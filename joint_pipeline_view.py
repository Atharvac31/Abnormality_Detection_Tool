import numpy as np
import torch
import cv2
from scipy.signal import hilbert, find_peaks
from scipy.stats import skew, kurtosis

FEATURE_ORDER = [
    "max","mean","std","ptv","skew","kurtosis",
    "spike_count","rms","energy","fft_ratio",
    "env_mean","env_std","env_growth",
    "slope","norm_slope","zero_crossings","peak_density"
]



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



def run_joint_pipeline(img, model, scaler, device):

    # =============================
    # 1. SIGNAL
    # =============================
    signal = extract_graph_points(img)
    signal = signal - np.min(signal)
    signal = cv2.GaussianBlur(signal.reshape(-1,1), (5,1), 0).flatten()

    # =============================
    # 2. FEATURES
    # =============================
    features = extract_features(signal)

    # 🔥 Feature Insights (NEW)
    feature_insights = {
        "signal_behavior": "Stable" if features["std"] < 0.2 else "Highly varying",
        "spike_analysis": "Frequent spikes" if features["spike_count"] > 10 else "Low spikes",
        "frequency_pattern": "Dominant frequency" if features["fft_ratio"] > 0.3 else "No strong frequency"
    }

    # =============================
    # 3. BASELINE MODEL
    # =============================
    feature_values = np.array(
        [features.get(f, 0.0) for f in FEATURE_ORDER],
        dtype=np.float32
    ).reshape(1, -1)

    x_scaled = scaler.transform(feature_values)
    x = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)

    baseline_pred = int(pred.item())
    baseline_conf = float(conf.item())

    # 🔥 Top Feature Importance (NEW)
    importance = np.abs(feature_values[0])
    top_idx = np.argsort(importance)[-5:][::-1]

    top_features = [
        {
            "name": FEATURE_ORDER[i],
            "value": float(feature_values[0][i])
        }
        for i in top_idx
    ]

    # =============================
    # 4. WEAK LABELS
    # =============================
    thresholds = {
        "high_variance": 0.25,
        "high_spikes": 10,
        "high_energy": 0.4,
        "high_fft": 0.3
    }

    lf_outputs = {
        "high_variance": features["std"],
        "high_spikes": features["spike_count"],
        "high_energy": features["energy"],
        "high_fft": features["fft_ratio"]
    }

    weak_labels = {
        key: int(lf_outputs[key] > thresholds[key])
        for key in thresholds
    }

    weak_score = np.mean(list(weak_labels.values()))

    # =============================
    # 5. JOINT DECISION
    # =============================
    weak_weight = 0.4
    model_weight = 0.6

    joint_score = weak_weight * weak_score + model_weight * baseline_conf

    # 🔥 Conflict Detection (NEW)
    conflict = (
        (baseline_conf > 0.8 and weak_score < 0.25) or
        (baseline_conf < 0.4 and weak_score > 0.75)
    )

    # 🔥 Decision Logic (IMPROVED)
    if baseline_conf > 0.9 and weak_score < 0.2:
        final_pred = "MODEL_OVERRIDE_ABNORMAL"
    elif joint_score > 0.7:
        final_pred = "STRONGLY ABNORMAL"
    elif joint_score > 0.55:
        final_pred = "ABNORMAL"
    elif joint_score < 0.35:
        final_pred = "NORMAL"
    else:
        final_pred = "UNCERTAIN"

    # 🔥 Reasoning (NEW)
    if conflict:
        reasoning = "Conflict between model and rules detected"
    elif baseline_conf > weak_score:
        reasoning = "Model influenced decision more"
    else:
        reasoning = "Weak rules influenced decision more"

    # =============================
    # RETURN (ADMIN VIEW FORMAT)
    # =============================
    return {
        "steps": {

            # =============================
            # FEATURE EXTRACTION
            # =============================
            "feature_extraction": {
                "values": features,
                "insights": feature_insights
            },

            # =============================
            # WEAK LABELING
            # =============================
            "weak_labeling": {
                "lf_outputs": lf_outputs,
                "thresholds": thresholds,
                "weak_labels": weak_labels,
                "score": float(weak_score)
            },

            # =============================
            # BASELINE MODEL
            # =============================
            "baseline_model": {
                "prediction": baseline_pred,
                "confidence": baseline_conf,
                "top_features": top_features,
                "explanation": "Prediction based on combined signal patterns and feature interactions"
            },

            # =============================
            # SUBSET SELECTION
            # =============================
            "subset_selection": {
                "selected_samples": 0,
                "method": "SPEAR (not implemented)",
                "strategy": "Selects diverse and uncertain samples",
                "benefit": "Improves training efficiency"
            },

            # =============================
            # JOINT LEARNING
            # =============================
            "joint_learning": {
                "joint_score": float(joint_score),
                "final_prediction": final_pred,
                "weak_score": float(weak_score),
                "model_score": float(baseline_conf),
                "weak_contribution": weak_weight,
                "model_contribution": model_weight,
                "conflict": conflict,
                "reasoning": reasoning
            }
        }
    }