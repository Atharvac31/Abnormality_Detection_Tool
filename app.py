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

# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        print("⚠️ Fallback: extracting weak signal")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pts = []
        for x in range(gray.shape[1]):
            column = gray[:, x]
            y = np.argmin(column)  # darkest pixel
            pts.append((x, y))

        return np.array(pts)

    return pts[np.argsort(pts[:, 0])]


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

    RULE_EXPLANATIONS = {
        "Asymmetric Signal": {
            "simple": "The signal is strongly biased to one side instead of being balanced.",
            "impact": "This usually indicates irregular or impulsive system behavior."
        },
        "Heavy-Tailed": {
            "simple": "The signal contains frequent extreme peaks.",
            "impact": "This often points to shocks, impacts, or abnormal events."
        },
        "Mechanical Resonance": {
            "simple": "One repeating frequency dominates the signal.",
            "impact": "This suggests resonance or mechanical instability."
        },
        "High Peak": {
            "simple": "The signal reaches unusually high values.",
            "impact": "This may indicate overload or excessive stress."
        },
        "Large Fluctuation": {
            "simple": "The signal swings widely between high and low values.",
            "impact": "This reflects unstable or poorly controlled behavior."
        },
        "Growing Instability": {
            "simple": "The signal amplitude increases over time.",
            "impact": "This is often an early warning sign of failure."
        }
    }

    RULE_WEIGHTS = {
    "Mechanical Resonance": 0.15,
    "Heavy-Tailed": 0.12,
    "Asymmetric Signal": 0.10,
    "High Peak": 0.10,
    "High RMS Energy": 0.10,
    "Large Fluctuation": 0.07,
    "Highly Variable": 0.06,
    "Spiky Pattern": 0.07,
    "Envelope Growth": 0.10,
    "Oscillatory Nature": 0.05,
    "Strong Trend": 0.05
}


    # --------------------------------------------------
    # Extract graph points
    # --------------------------------------------------
    pts = extract_graph_points(img)
    x_px, y_px = pts[:, 0], pts[:, 1].astype(float)

    # --------------------------------------------------
    # SIGNAL CONSTRUCTION
    # --------------------------------------------------
    if graph_type == "scaled":
        c = CALIBRATION
        x = c["x_min"] + (x_px - c["x_left"]) * (c["x_max"] - c["x_min"]) / (c["x_right"] - c["x_left"])
        signal = c["y_min"] + (c["y_bottom"] - y_px) * (c["y_max"] - c["y_min"]) / (c["y_bottom"] - c["y_top"])
        analysis_mode = "PHYSICAL"

    else:
        x = np.arange(len(y_px))
        y_min = user_cfg.get("y_min", type=float) if hasattr(user_cfg.get("y_min"), "__float__") else user_cfg.get("y_min")
        y_max = user_cfg.get("y_max", type=float) if hasattr(user_cfg.get("y_max"), "__float__") else user_cfg.get("y_max")

        if y_min is not None and y_max is not None:
            signal = y_min + (y_px.max() - y_px) * (y_max - y_min) / (y_px.max() - y_px.min() + 1e-8)
            analysis_mode = "CALIBRATED"
        else:
            signal = (y_px - y_px.min()) / (y_px.max() - y_px.min() + 1e-8)
            has_context = any(user_cfg.get(k) for k in ["sensitivity", "expected_behavior", "x_axis_meaning"])
            analysis_mode = "RELATIVE" if has_context else "BASIC"

    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------
    features = {
        "max": float(np.max(signal)),
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "ptv": float(np.ptp(signal)),
        "skew": float(skew(signal)),
        "kurtosis": float(kurtosis(signal, fisher=False)),
    }
    # ===============================
# ADDITIONAL FEATURES (ML REQUIRED)
# ===============================

# RMS
    features["rms"] = float(np.sqrt(np.mean(signal ** 2)))

# Energy
    features["energy"] = float(np.sum(signal ** 2))

# FFT ratio (already computed)
    features["fft_ratio"] = float(fft_ratio)

# Envelope stats
    features["env_mean"] = float(np.mean(envelope))
    features["env_std"] = float(np.std(envelope))
    features["env_growth"] = float(env_growth_ratio)

# Trend features
    slope = np.polyfit(range(len(signal)), signal, 1)[0]
    features["slope"] = float(slope)
    features["norm_slope"] = float(slope / (np.std(signal) + 1e-8))

# Zero crossings
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    features["zero_crossings"] = int(len(zero_crossings))

# Peak density
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal)
    features["peak_density"] = float(len(peaks) / (len(signal) + 1e-8))
    # ✅ Improved spike detection (robust)
    diffs = np.abs(np.diff(signal))
    spike_thresh = max(
        np.mean(diffs) + 3 * np.std(diffs),
        np.percentile(diffs, 95)
    )
    features["spike_count"] = int(np.sum(diffs > spike_thresh))

    # ✅ Stable envelope growth
    envelope = np.abs(hilbert(signal)) if len(signal) >= 10 else np.zeros_like(signal)
    env_slope = np.polyfit(range(len(envelope)), envelope, 1)[0] if len(signal) >= 10 else 0.0
    mean_env = np.mean(envelope)
    env_growth_ratio = env_slope / (mean_env + 1e-6)

    # ✅ FFT energy ratio
    fft_energy = np.abs(fft(signal)) ** 2
    fft_ratio = np.max(fft_energy) / (np.sum(fft_energy) + 1e-8)

    # SAFE FFT
    if len(signal) > 5:
        fft_energy = np.abs(fft(signal)) ** 2
        fft_ratio = np.max(fft_energy) / (np.sum(fft_energy) + 1e-8)
    else:
        fft_ratio = 0.0
    # --------------------------------------------------
    # RULE REPORTING + EXPLAINABILITY
    # --------------------------------------------------
    rule_report = []
    explanations = []
    counterfactuals = []
    contributions = []

    def violation_factor(value, limit):
        """Severity scaling (caps extreme influence)"""
        return min(abs(value) / (abs(limit) + 1e-8), 2.0)

    def add_rule(name, value, threshold, abnormal, limit_value=None):
        rule_report.append({
            "rule": name,
            "value": round(float(value), 4),
            "threshold": threshold,
            "status": "ABNORMAL" if abnormal else "NORMAL"
        })

        weight = RULE_WEIGHTS.get(name, 1.0)

        if abnormal:
            info = RULE_EXPLANATIONS.get(name, {})

            explanations.append({
                "rule": name,
                "what_happened": info.get("simple", f"{name} exceeded normal limits."),
                "why_it_matters": info.get("impact", "This is associated with abnormal system behavior."),
                "measured_value": round(float(value), 4),
                "expected_limit": threshold
            })

            counterfactuals.append(
                f"If {name.lower()} is corrected, the system would move closer to normal behavior."
            )

            severity_scale = violation_factor(value, limit_value) if limit_value else 1.0

            contributions.append({
                "rule": name,
                "impact": round(weight * severity_scale, 3)
            })

    # --------------------------------------------------
    # RULE SETS
    # --------------------------------------------------
    if analysis_mode == "PHYSICAL":
        add_rule("High Peak", features["max"], "> 25", features["max"] > 25, 25)
        rms = np.sqrt(np.mean(signal ** 2))
        add_rule("High RMS Energy", rms, "> 15", rms > 15, 15)
        add_rule("Large Fluctuation", features["ptv"], "> 20", features["ptv"] > 20, 20)
        add_rule("Highly Variable", features["std"], "> 8", features["std"] > 8, 8)
        add_rule("Asymmetric Signal", features["skew"], "|skew| > 1", abs(features["skew"]) > 1, 1)
        add_rule("Heavy-Tailed", features["kurtosis"], "> 5", features["kurtosis"] > 5, 5)
        add_rule("Mechanical Resonance", fft_ratio, "> 0.8", fft_ratio > 0.8, 0.8)
        add_rule("Growing Instability", env_growth_ratio, "> 0.1", env_growth_ratio > 0.1, 0.1)

    elif analysis_mode == "CALIBRATED":
        span = y_max - y_min
        add_rule("Large Fluctuation", features["ptv"] / span, "> 0.6", features["ptv"] / span > 0.6, 0.6)
        add_rule("Mechanical Resonance", fft_ratio, "> 0.8", fft_ratio > 0.8, 0.8)
        add_rule("Envelope Growth", env_growth_ratio, "> 0.1", env_growth_ratio > 0.1, 0.1)

    elif analysis_mode == "RELATIVE":
        sensitivity = user_cfg.get("sensitivity", "medium")
        spike_map = {"low": 8, "medium": 5, "high": 3}
        ptv_map = {"low": 6.0, "medium": 4.5, "high": 3.5}

        add_rule("Spiky Pattern", features["spike_count"],
                 f"≥ {spike_map[sensitivity]}",
                 features["spike_count"] >= spike_map[sensitivity],
                 spike_map[sensitivity])

        add_rule("Large Fluctuation", features["ptv"] / features["std"],
                 f"> {ptv_map[sensitivity]}",
                 features["ptv"] / (features["std"] + 1e-8) > ptv_map[sensitivity],
                 ptv_map[sensitivity])

        add_rule("Mechanical Resonance", fft_ratio, "> 0.8", fft_ratio > 0.8, 0.8)
        add_rule("Envelope Growth", env_growth_ratio, "> 0.1", env_growth_ratio > 0.1, 0.1)

    else:  # BASIC
        slope = np.polyfit(x, signal, 1)[0]
        norm_slope = abs(slope) / (np.std(signal) + 1e-8)

        add_rule("Oscillatory Nature", fft_ratio, "> 0.8", fft_ratio > 0.8, 0.8)
        add_rule("Strong Trend", norm_slope, "> 0.2", norm_slope > 0.2, 0.2)

    # --------------------------------------------------
    # ✅ SEVERITY-AWARE WEIGHTED AGGREGATION
    # --------------------------------------------------
    risk_score = 0.0
    max_score = 0.0

    for rule in rule_report:
        weight = RULE_WEIGHTS.get(rule["rule"], 1.0)
        max_score += weight
        if rule["status"] == "ABNORMAL":
            risk_score += weight

    normalized_risk = risk_score / (max_score + 1e-8)

    if normalized_risk >= 0.6:
        status = "STRONGLY ABNORMAL"
    elif normalized_risk >= 0.25:
        status = "ABNORMAL"
    else:
        status = "NORMAL"

    confidence = normalized_risk

    if analysis_mode == "RELATIVE":
        confidence *= 0.75
    elif analysis_mode == "BASIC":
        confidence *= 0.5

    reliability = {
        "PHYSICAL": 1.0,
        "CALIBRATED": 0.9,
        "RELATIVE": 0.75,
        "BASIC": 0.5
    }[analysis_mode]

    severity = {
        "STRONGLY ABNORMAL": "HIGH",
        "ABNORMAL": "MEDIUM",
        "NORMAL": "LOW"
    }[status]

    primary_cause = (
        max(contributions, key=lambda x: x["impact"])["rule"]
        if contributions else None
    )

    slope = np.polyfit(x, signal, 1)[0]
    norm_slope = slope / (np.std(signal) + 1e-8)

    trend = "Increasing" if norm_slope > 0.2 else "Decreasing" if norm_slope < -0.2 else "Steady"
    if fft_ratio > 0.8:
        trend += ", Cyclic"

    top_rules = [r["rule"] for r in rule_report if r["status"] == "ABNORMAL"][:3]

    if not top_rules:
        summary = "The signal appears stable and within normal operating conditions."
    else:
        joined = ", ".join(top_rules)
        summary = f"The system shows {severity.lower()} abnormal behavior mainly due to {joined.lower()}."

    return {
        "analysis_mode": analysis_mode,
        "features": features,
        "abnormal_rules": rule_report,
        "status": status,
        "severity": severity,
        "confidence": round(confidence, 2),
        "reliability": reliability,
        "trend": trend,
        "summary": summary,
        "primary_cause": primary_cause,
        "explanations": explanations,
        "counterfactuals": counterfactuals,
        "risk_score": round(risk_score, 2),
        "normalized_risk": round(normalized_risk, 2),
        "rule_contributions": sorted(contributions, key=lambda x: x["impact"], reverse=True)
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
