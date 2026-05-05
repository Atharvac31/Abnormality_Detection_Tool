import os
import re
import signal
import cv2
import numpy as np
from pygit2 import features
import pytesseract
from flask import Flask, request, jsonify, render_template
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, hilbert
from scipy.fft import fft
from streamlit import status
from models.cnn_extractor import CNNFeatureExtractor
from models.cage_aggregator import SimpleCAGE
# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
cnn_extractor = CNNFeatureExtractor()
cage = SimpleCAGE(lf_acc_prior=0.75)
# ============================================================
# CALIBRATION (USED ONLY IF GRAPH IS SCALED)
# ============================================================

CALIBRATION = {
    "x_left": 80,
    "x_right": 740,
    "x_min": 0,
    "x_max": 100,
    "y_top": 120,
    "y_bottom": 520,
    "y_min": 3.8,
    "y_max": 5.4
}

# ============================================================
# GRAPH TYPE DETECTION (SCALED VS UNSCALED)
# ============================================================

def detect_scaled_graph(img):
    h, w, _ = img.shape
    left = img[:, :int(0.3 * w)]
    bottom = img[int(0.7 * h):, :]

    def ocr(region):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return pytesseract.image_to_string(gray, config="--psm 6").lower()

    text = ocr(left) + " " + ocr(bottom)

    unit_keywords = [
        "mm/s", "m/s", "hz", "rpm",
        "vibration", "velocity", "acceleration",
        "amplitude", "pressure", "temperature", "voltage"
    ]

    if any(u in text for u in unit_keywords):
        return True

    numbers = re.findall(r"\d+\.\d+|\d+", text)
    return len(numbers) >= 3


# ============================================================
# GRAPH EXTRACTION
# ============================================================

def get_curve_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def extract_by_columns(mask):
    pts = []
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys):
            pts.append((x, int(np.median(ys))))
    return np.array(pts)


def extract_by_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        if cv2.contourArea(c) > 5:
            M = cv2.moments(c)
            if M["m00"]:
                pts.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    return np.array(pts)


def extract_graph_points(img):
    mask = get_curve_mask(img)
    col_pts = extract_by_columns(mask)
    pts = col_pts if len(col_pts) > img.shape[1] * 0.2 else extract_by_contours(mask)

    if len(pts) < 5:
        raise ValueError("Graph extraction failed")

    return pts[np.argsort(pts[:, 0])]

def adapt_threshold(physical_thresh, analysis_mode):
    """
    Adjust thresholds for RELATIVE/BASIC signals
    """
    if analysis_mode in ["RELATIVE", "BASIC"]:
        # Map rough physical → normalized equivalents
        mapping = {
            25: 0.8,
            20: 0.6,
            15: 0.5,
            8: 0.25,
            5: 0.15,
            0.2: 0.2,
            0.1: 0.1
        }
        return mapping.get(physical_thresh, physical_thresh)
    return physical_thresh
def adaptive_scale(analysis_mode, physical_scale, relative_scale):
    return relative_scale if analysis_mode in ["RELATIVE", "BASIC"] else physical_scale

def convert_lfs_to_weak_labels(lf_outputs):

    weak_labels = []

    for lf in lf_outputs.values():

        p = lf["prob_abnormal"]

        if p > 0.6:
            weak_labels.append(1)

        elif p < 0.4:
            weak_labels.append(0)

        else:
            weak_labels.append(-1)

    return weak_labels
# ============================================================
# MAIN ANALYSIS
# ============================================================
def analyze_graph(image_path, graph_type, user_cfg=None):

    import numpy as np
    import cv2
    from scipy.stats import skew, kurtosis
    from scipy.signal import hilbert
    from scipy.fft import fft

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image")

    user_cfg = user_cfg or {}

    # ==================================================
    # CNN EMBEDDING
    # ==================================================

    cnn_embedding = cnn_extractor.extract(image_path)

# 🔥 FORCE NUMPY ARRAY
    cnn_embedding = np.array(cnn_embedding, dtype=np.float32)
    if cnn_embedding.ndim == 0:
        cnn_embedding = np.array([cnn_embedding], dtype=np.float32)
    print("DEBUG CNN TYPE:", type(cnn_embedding))
    print("DEBUG CNN SHAPE:", cnn_embedding.shape)
    # ==================================================
    # GRAPH EXTRACTION
    # ==================================================

    pts = extract_graph_points(img)
    x_px, y_px = pts[:, 0], pts[:, 1].astype(float)

    # ==================================================
    # SIGNAL CONSTRUCTION
    # ==================================================

    if graph_type == "scaled":

        c = CALIBRATION

        x = c["x_min"] + (x_px - c["x_left"]) * (c["x_max"] - c["x_min"]) / (
            c["x_right"] - c["x_left"]
        )

        signal = c["y_min"] + (c["y_bottom"] - y_px) * (c["y_max"] - c["y_min"]) / (
            c["y_bottom"] - c["y_top"]
        )

        analysis_mode = "PHYSICAL"

    else:

        x = np.arange(len(y_px))

        y_min = user_cfg.get("y_min")
        y_max = user_cfg.get("y_max")

        if y_min is not None and y_max is not None:

            signal = y_min + (y_px.max() - y_px) * (y_max - y_min) / (
                y_px.max() - y_px.min() + 1e-8
            )

            analysis_mode = "CALIBRATED"

        else:

            signal = (y_px - y_px.min()) / (y_px.max() - y_px.min() + 1e-8)

            has_context = any(
                user_cfg.get(k)
                for k in ["sensitivity", "expected_behavior", "x_axis_meaning"]
            )

            analysis_mode = "RELATIVE" if has_context else "BASIC"

    # ==================================================
    # FEATURE EXTRACTION
    # ==================================================

    features = {}

    features["max"] = float(np.max(signal))
    features["mean"] = float(np.mean(signal))
    features["std"] = float(np.std(signal))
    features["variance"] = float(np.var(signal))
    features["ptv"] = float(np.ptp(signal))
    features["skew"] = float(skew(signal))
    features["kurtosis"] = float(kurtosis(signal, fisher=False))

    # CNN features
    features["cnn_mean"] = float(np.mean(cnn_embedding))
    features["cnn_std"] = float(np.std(cnn_embedding))
    features["cnn_energy"] = float(np.linalg.norm(cnn_embedding))

    # ==================================================
    # SPIKE DETECTION (MAD BASED)
    # ==================================================

    diffs = np.abs(np.diff(signal))

    mad = np.median(np.abs(diffs - np.median(diffs)))

    spike_thresh = np.median(diffs) + 5 * mad

    features["spike_count"] = int(np.sum(diffs > spike_thresh))

    # ==================================================
    # ENVELOPE INSTABILITY
    # ==================================================

    envelope = np.abs(hilbert(signal))

    third = len(envelope) // 3

    env_start = np.mean(envelope[:third])
    env_end = np.mean(envelope[-third:])

    env_growth_ratio = (env_end - env_start) / (env_start + 1e-6)

    features["env_growth_ratio"] = float(env_growth_ratio)

    # ==================================================
    # FFT RESONANCE
    # ==================================================

    fft_energy = np.abs(fft(signal)) ** 2
    fft_energy[0] = 0

    dominant = np.max(fft_energy)
    total = np.sum(fft_energy)

    fft_ratio = dominant / (total + 1e-8)

    features["fft_ratio"] = float(fft_ratio)

    # ==================================================
    # PLATEAU DETECTION
    # ==================================================

    plateau_ratio = np.mean(np.abs(np.diff(signal)) < 0.01 * np.std(signal))

    features["plateau_ratio"] = float(plateau_ratio)

    # ==================================================
    # LABELING FUNCTIONS
    # ==================================================

    def soft_score(value, threshold, scale=1.0):

        return 1 / (1 + np.exp(-(value - threshold) / scale))

    def confidence_from_distance(value, threshold, max_range):

        return min(abs(value - threshold) / (max_range + 1e-8), 1.0)

    lf_outputs = {}

    def add_lf(name, value, threshold, scale, max_range):

        prob = soft_score(value, threshold, scale)

        conf = confidence_from_distance(value, threshold, max_range)

        lf_outputs[name] = {
            "prob_abnormal": float(prob),
            "confidence": float(conf),
        }

    # ==================================================
    # RULES
    # ==================================================

    add_lf("High Peak", features["max"], adapt_threshold(25, analysis_mode), 5, 25)

    add_lf("Large Fluctuation", features["ptv"], adapt_threshold(20, analysis_mode), 4, 20)

    add_lf("Highly Variable", features["std"], adapt_threshold(8, analysis_mode), 2, 8)

    add_lf("Asymmetric Signal", abs(features["skew"]), 1, 0.5, 2)

    add_lf("Heavy Tailed", features["kurtosis"], 5, 1.5, 10)

    add_lf("Mechanical Resonance", features["fft_ratio"], 0.7, 0.1, 1)

    add_lf("Growing Instability", features["env_growth_ratio"], 0.1, 0.03, 0.3)

    add_lf("Spiky Pattern", features["spike_count"], 6, 2, 15)

    add_lf("Mean Abnormal", features["mean"], adapt_threshold(20, analysis_mode), 3, 20)

    add_lf("Plateau", features["plateau_ratio"], 0.1, 0.03, 0.5)

    add_lf("CNN Complexity", features["cnn_energy"], 20, 5, 20)

    add_lf("CNN Variability", features["cnn_std"], 0.8, 0.2, 1.5)

    # ==================================================
    # WEAK LABEL MATRIX
    # ==================================================

    weak_labels = convert_lfs_to_weak_labels(lf_outputs)

    # ==================================================
    # CAGE AGGREGATION
    # ==================================================

    abnormal_prob = cage.predict(weak_labels,lf_outputs)

    # ==================================================
    # STATUS LOGIC
    # ==================================================

    if abnormal_prob > 0.8:
        status = "STRONGLY ABNORMAL"

    elif abnormal_prob > 0.55:
        status = "ABNORMAL"

    elif abnormal_prob < 0.35:
        status = "NORMAL"

    else:
        status = "UNCERTAIN"

    # ==================================================
    # CONFIDENCE
    # ==================================================

    confs = [lf["confidence"] for lf in lf_outputs.values()]

    confidence = float(np.mean(confs))

    # ==================================================
    # RETURN
    # ==================================================

    return {
        "analysis_mode": analysis_mode,
        "features": features,
        "lf_outputs": lf_outputs,
        "weak_labels": weak_labels,
        "abnormal_probability": float(abnormal_prob),
        "status": status,
        "confidence": confidence,
    }

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect-graph-type", methods=["POST"])
def detect_graph_type():
    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)
    graph_type = "scaled" if detect_scaled_graph(img) else "unscaled"

    return jsonify({"graph_type": graph_type})

@app.route("/analyze", methods=["POST"])
def analyze():
    # --------------------------------------------------
    # 1. Validate image
    # --------------------------------------------------
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file or file.filename.strip() == "":
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    img = cv2.imread(save_path)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # --------------------------------------------------
    # 2. Detect graph type
    # --------------------------------------------------
    graph_type = "scaled" if detect_scaled_graph(img) else "unscaled"

    # --------------------------------------------------
    # 3. Build user_cfg ONLY from provided fields
    #    (ignore empty values)
    # --------------------------------------------------
    user_cfg = {}

    def add_if_present(key):
        val = request.form.get(key)
        if val is not None and val.strip() != "":
            user_cfg[key] = val

    add_if_present("y_axis_meaning")
    add_if_present("x_axis_meaning")
    add_if_present("expected_behavior")
    add_if_present("sensitivity")

    # --------------------------------------------------
    # 4. Analyze (no blocking, adaptive logic inside)
    # --------------------------------------------------
    try:
        result = analyze_graph(
            image_path=save_path,
            graph_type=graph_type,
            user_cfg=user_cfg
        )

        result["graph_type"] = graph_type
        result["filename"] = file.filename

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "Graph analysis failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
