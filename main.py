import os
import cv2
import torch
import numpy as np
import pytesseract
import re
import uuid
import joblib

from flask import Flask, request, jsonify, render_template
import torch.nn as nn
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, hilbert
from scipy.fft import fft
from lfs import analyze_graph
from joint_pipeline_view import run_joint_pipeline
# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# LOAD MODEL + SCALER
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = joblib.load("scaler.pkl")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(17, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

model = MLP()
model.load_state_dict(torch.load("joint_learning_mlp.pt", map_location=device))
model.to(device)
model.eval()

print("✅ Model Loaded")

# ============================================================
# GRAPH TYPE DETECTION
# ============================================================
FEATURE_ORDER = [
    "max","mean","std","ptv","skew","kurtosis",
    "spike_count","rms","energy","fft_ratio",
    "env_mean","env_std","env_growth",
    "slope","norm_slope","zero_crossings","peak_density"
]
def detect_scaled_graph(img):
    h, w, _ = img.shape
    left = img[:, :int(0.3 * w)]
    bottom = img[int(0.7 * h):, :]

    def ocr(region):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        return pytesseract.image_to_string(gray, config="--psm 6").lower()

    text = ocr(left) + " " + ocr(bottom)

    if any(u in text for u in ["mm/s", "hz", "rpm", "voltage"]):
        return "scaled"

    numbers = re.findall(r"\d+\.\d+|\d+", text)
    return "scaled" if len(numbers) >= 3 else "unscaled"

# ============================================================
# GRAPH EXTRACTION
# ============================================================

def extract_graph_points(img):

    if img is None:
        raise ValueError("Image not loaded")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 Step 1: edge detect
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape
    signal = []

    for x in range(w):

        ys = np.where(edges[:, x] > 0)[0]

        if len(ys) > 0:
            # ✅ take MEDIAN (not topmost)
            y = int(np.median(ys))
            signal.append(h - y)
        else:
            signal.append(signal[-1] if signal else 0)

    signal = np.array(signal, dtype=np.float32)

    # 🔥 Step 2: normalize scale (CRITICAL)
    signal = signal - np.min(signal)
    signal = signal / (np.max(signal) + 1e-8)

    # 🔥 Step 3: smooth
    signal = cv2.GaussianBlur(signal.reshape(-1,1), (7,1), 0).flatten()

    return signal

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.item()
    else:
        return obj


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(signal):
    signal = np.nan_to_num(signal)

    if np.std(signal) < 1e-6:
        signal = signal + 1e-6

    features = {}

    # basic
    features["max"] = float(np.max(signal))
    features["mean"] = float(np.mean(signal))
    features["std"] = float(np.std(signal))
    features["ptv"] = float(np.ptp(signal))
    features["skew"] = float(skew(signal))
    features["kurtosis"] = float(kurtosis(signal, fisher=False))
    fft_vals = np.abs(np.fft.fft(signal))

    if np.sum(fft_vals) == 0:
        fft_ratio = 0.0
    else:
        fft_ratio = float(np.max(fft_vals) / (np.sum(fft_vals) + 1e-8))

    features["fft_ratio"] = fft_ratio
    

    # spike
    # spike count (RAW, not normalized)
    diffs = np.abs(np.diff(signal))
    threshold = 2.5 * np.std(signal)
    features["spike_count"] = int(np.sum(diffs > threshold))

# rms / energy (RAW scale)
    features["rms"] = float(np.sqrt(np.mean(signal ** 2)))
    features["energy"] = float(np.sum(signal ** 2) / len(signal))  # FIXED

# envelope (REMOVE normalization)
    envelope = np.abs(hilbert(signal))
    features["env_mean"] = float(np.mean(envelope))
    features["env_std"] = float(np.std(envelope))

    env_slope = np.polyfit(range(len(envelope)), envelope, 1)[0]
    features["env_growth"] = float(env_slope)

    # trend
    slope = np.polyfit(range(len(signal)), signal, 1)[0]
    features["slope"] = float(slope)
    features["norm_slope"] = float(slope / (np.std(signal) + 1e-8))

    # zero crossing
    features["zero_crossings"] = int(len(np.where(np.diff(np.sign(signal)))[0]))

    # peaks
    peaks, _ = find_peaks(signal)
    features["peak_density"] = float(len(peaks) / len(signal))
    
    # Ensure all expected features exist
    for key in FEATURE_ORDER:
        features.setdefault(key, 0.0)
    
    return features

    
# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(features):

    x = np.array([features.get(f, 0.0) for f in FEATURE_ORDER], dtype=np.float32)
    x = x.reshape(1, -1)

    # ✅ Apply SAME scaler as training
    x = torch.tensor(x, dtype=torch.float32).to(device)

    return x

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/pipeline")
def pipeline_view():
    return render_template("pipeline.html")

@app.route("/pipeline-joint", methods=["POST"])
def pipeline_joint():
    try:
        file = request.files["image"]

        path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + file.filename)
        file.save(path)

        img = cv2.imread(path)

        result = run_joint_pipeline(img, model, scaler, device)

        return jsonify(make_json_safe(result))

    except Exception as e:
        print("🔥 PIPELINE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/detect-graph-type", methods=["POST"])
def detect_graph_type():
    file = request.files["image"]

    path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + file.filename)
    file.save(path)

    img = cv2.imread(path)
    return jsonify({"graph_type": detect_scaled_graph(img)})

@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + file.filename)
    file.save(path)

    img = cv2.imread(path)

    graph_type = detect_scaled_graph(img)

    # -------------------
    # Extract signal
    # -------------------
    signal = extract_graph_points(img)
    signal = signal - np.min(signal)
    signal = cv2.GaussianBlur(signal.reshape(-1,1), (5,1), 0).flatten()

    # -------------------
    # Features
    # -------------------
    features = extract_features(signal)

    # -------------------
    # Model
    # -------------------
    feature_values = np.array(
    [features.get(f, 0.0) for f in FEATURE_ORDER],
    dtype=np.float32
    ).reshape(1, -1)

    x_scaled = scaler.transform(feature_values)
    x = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x)
        temperature = 2.0
        probs = torch.softmax(out / temperature, dim=1)
        conf, pred = torch.max(probs, dim=1)

    print("\n=== DEBUG ===")
    print("Features:", features)
    print("Input to model:", x.cpu().numpy())
    print("STD:", features["std"])
    print("ENERGY:", features["energy"])
    
    return jsonify(make_json_safe({
    "prediction": int(pred.item()),
    "confidence": float(conf.item()),
    "features": features,
    "graph_type": graph_type
}))
# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)