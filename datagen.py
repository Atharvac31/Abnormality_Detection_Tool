import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

from scipy.signal import hilbert, find_peaks

# ===============================
# CONFIG
# ===============================

NUM_SAMPLES = 500
SIGNAL_LENGTH = 100

OUTPUT_DIR = "synthetic_dataset_v2"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "labels.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# ===============================
# SIGNAL GENERATION (UNCHANGED)
# ===============================

def generate_ultra_complex_signal(length=100):
    t = np.linspace(0, 1, length)
    signal = np.zeros(length)

    base_amp = np.random.uniform(0.5, 8)
    num_freq = np.random.randint(2, 6)

    for _ in range(num_freq):
        freq = np.random.uniform(0.5, 15)
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(0.2, 1.5)
        signal += amp * np.sin(2*np.pi*freq*t + phase)

    signal *= base_amp

    envelope = 1 + np.random.uniform(0.3, 1.2) * np.sin(2*np.pi*np.random.uniform(0.2,4)*t)
    signal *= envelope

    signal += np.random.uniform(-5,5) * t

    if np.random.rand() < 0.5:
        signal = np.tanh(signal)

    signal += np.cumsum(np.random.normal(0,0.05,length))
    signal += np.random.normal(0,0.5,length)

    return signal


# ===============================
# 🔥 SIGNAL EXTRACTION (MATCH MAIN)
# ===============================

def extract_signal_from_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold to isolate line
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    signal = []

    for x in range(w):
        ys = np.where(thresh[:, x] > 0)[0]

        if len(ys) > 0:
            signal.append(h - np.mean(ys))
        else:
            signal.append(signal[-1] if signal else 0)

    signal = np.array(signal)

    # minimal processing only
    signal = signal - np.min(signal)
    signal = signal / (np.max(signal) + 1e-8)

    return signal


# ===============================
# 🔥 FEATURE EXTRACTION (MATCH MAIN)
# ===============================

def extract_features(signal):

    features = {}

    features["max"] = float(np.max(signal))
    features["mean"] = float(np.mean(signal))
    features["std"] = float(np.std(signal))
    features["ptv"] = float(np.ptp(signal))

    features["rms"] = float(np.sqrt(np.mean(signal**2)))
    features["energy"] = float(np.sum(signal**2) / len(signal))

    features["skew"] = float(pd.Series(signal).skew())
    features["kurtosis"] = float(pd.Series(signal).kurtosis())

    # spikes
    diff = np.abs(np.diff(signal))
    threshold = np.mean(diff) + 2*np.std(diff)
    features["spike_count"] = int(np.sum(diff > threshold))

    # fft
    fft_vals = np.abs(np.fft.fft(signal))
    features["fft_ratio"] = float(np.max(fft_vals) / (np.sum(fft_vals) + 1e-8))

    # envelope
    envelope = np.abs(hilbert(signal))
    features["env_mean"] = float(np.mean(envelope))
    features["env_std"] = float(np.std(envelope))

    env_slope = np.polyfit(range(len(envelope)), envelope, 1)[0]
    features["env_growth"] = float(env_slope)

    # slope
    slope = np.polyfit(range(len(signal)), signal, 1)[0]
    features["slope"] = float(slope)
    features["norm_slope"] = float(slope / (np.std(signal) + 1e-8))

    # zero crossings
    features["zero_crossings"] = int(np.sum(np.diff(np.sign(signal)) != 0))

    # peak density
    features["peak_density"] = float(features["spike_count"] / len(signal))

    return features


# ===============================
# DATASET GENERATION
# ===============================

rows = []

for i in range(NUM_SAMPLES):

    signal = generate_ultra_complex_signal(SIGNAL_LENGTH)

    abnormal = np.random.rand() < 0.4

    # apply faults
    if abnormal:

        fault_type = np.random.choice(["spikes", "noise", "drift", "drop"])

        if fault_type == "spikes":
            for _ in range(np.random.randint(5, 15)):
                idx = np.random.randint(0, len(signal))
                signal[idx] += np.random.uniform(5, 15)

        elif fault_type == "noise":
            signal += np.random.normal(0, 2, len(signal))

        elif fault_type == "drift":
            signal += np.linspace(0, np.random.uniform(5, 10), len(signal))

        elif fault_type == "drop":
            start = np.random.randint(10, 60)
            signal[start:start+10] *= 0.2

    img_path = os.path.join(IMG_DIR, f"img_{i:05d}.png")

    # save image
    plt.figure(figsize=(4,3))

# add jitter to line
    noisy_signal = signal + np.random.normal(0, 0.2, len(signal))

    plt.plot(noisy_signal, linewidth=np.random.uniform(1.5, 3))

# random styling
    plt.grid(np.random.rand() < 0.3)
    plt.title("")  

    plt.tight_layout()
    plt.savefig(img_path, dpi=100)
    plt.close()

# add image noise
    img = cv2.imread(img_path)
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    cv2.imwrite(img_path, img)

    # 🔥 IMPORTANT: reload image and extract signal
    img = cv2.imread(img_path)
    extracted_signal = extract_signal_from_image(img)

    # 🔥 IMPORTANT: extract features from extracted signal
    features = extract_features(extracted_signal)

    rows.append({
        "image_path": img_path,
        **features,
        "Abnormality": int(abnormal)
    })


# ===============================
# SAVE
# ===============================

df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)

print("✅ NEW DATASET GENERATED:", OUTPUT_DIR)