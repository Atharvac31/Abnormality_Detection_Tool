import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import hilbert, find_peaks

# ==========================
# CONFIGURATION
# ==========================

NUM_SAMPLES = 5000
SIGNAL_LENGTH = 100

OUTPUT_DIR = "unlabeled_vibration_dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "unlabeled_images")
CSV_PATH = os.path.join(OUTPUT_DIR, "data.csv")

os.makedirs(IMG_DIR, exist_ok=True)


# ==========================
# COMPLEX SIGNAL GENERATOR
# ==========================

def generate_signal(length=100):

    t = np.linspace(0,1,length)
    signal = np.zeros(length)

    # multi-frequency mixture
    for _ in range(np.random.randint(2,6)):

        amp = np.random.uniform(0.3,2)
        freq = np.random.uniform(0.5,15)
        phase = np.random.uniform(0,2*np.pi)

        signal += amp*np.sin(2*np.pi*freq*t + phase)

    # envelope modulation
    env = 1 + np.random.uniform(0.3,1.5)*np.sin(2*np.pi*np.random.uniform(0.2,3)*t)
    signal *= env

    # random trend
    signal += np.random.uniform(-4,4)*t

    # random walk drift
    signal += np.cumsum(np.random.normal(0,np.random.uniform(0.01,0.1),length))

    # gaussian noise
    signal += np.random.normal(0,np.random.uniform(0.1,1),length)

    # cavitation-like bursts
    if np.random.rand() < 0.5:

        for _ in range(np.random.randint(1,5)):

            pos = np.random.randint(5,length-5)
            width = np.random.randint(1,4)

            signal[pos:pos+width] += np.random.uniform(3,8)

    # plateau dropout
    if np.random.rand() < 0.3:

        start = np.random.randint(0,length-10)
        signal[start:start+10] = signal[start]

    # sudden jump
    if np.random.rand() < 0.3:

        pos = np.random.randint(20,length-20)
        signal[pos:] += np.random.uniform(-6,6)

    return signal


# ==========================
# FEATURE EXTRACTION
# ==========================

def compute_features(signal):

    t = np.arange(len(signal))

    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    var = np.var(signal)

    max_val = np.max(signal)
    min_val = np.min(signal)

    slope = np.polyfit(t,signal,1)[0]

    peaks,_ = find_peaks(signal)
    peak_count = len(peaks)

    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(len(signal),1)

    dominant_freq = fft_freqs[np.argmax(np.abs(fft_vals))]

    skew = pd.Series(signal).skew()
    kurt = pd.Series(signal).kurtosis()

    derivative = np.diff(signal)
    plateau_ratio = np.mean(np.abs(derivative) < 1e-3)

    envelope = np.abs(hilbert(signal))

    return {
        "Mean":mean,
        "Std":std,
        "RMS":rms,
        "Variance":var,
        "Max":max_val,
        "Min":min_val,
        "Slope":slope,
        "Peak_Count":peak_count,
        "Dominant_Freq":dominant_freq,
        "Skewness":skew,
        "Kurtosis":kurt,
        "Plateau_Ratio":plateau_ratio
    }


# ==========================
# DATASET GENERATION
# ==========================

rows = []

for i in range(NUM_SAMPLES):

    signal = generate_signal(SIGNAL_LENGTH)

    features = compute_features(signal)

    # machine context
    load = np.random.uniform(10,120)
    temp = np.random.uniform(10,120)
    rpm = np.random.uniform(500,6000)
    humidity = np.random.uniform(0,100)
    age = np.random.uniform(100,20000)

    img_path = os.path.join(IMG_DIR,f"img_{i:05d}.png")

    # randomized plot style
    plt.figure(figsize=(np.random.uniform(3,5),np.random.uniform(2.5,4)))

    plt.plot(
        signal,
        linewidth=np.random.uniform(0.5,3),
        linestyle=np.random.choice(["-","--","-.",":"]),
        color=np.random.choice(["blue","navy","red","green","black","purple"])
    )

    plt.grid(alpha=np.random.uniform(0.1,0.8))

    if np.random.rand() < 0.3:
        plt.xticks([])

    if np.random.rand() < 0.3:
        plt.yticks([])

    plt.title(np.random.choice([
        "Sensor Signal",
        "Vibration Signal",
        "Machine Monitoring",
        "Sensor Data"
    ]))

    plt.ylim(np.min(signal)-2,np.max(signal)+2)

    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    rows.append({
        "image_path":img_path,
        **features,
        "Load":load,
        "Temp":temp,
        "RPM":rpm,
        "Humidity":humidity,
        "Age":age
    })


# ==========================
# SAVE CSV
# ==========================

df = pd.DataFrame(rows)
df.to_csv(CSV_PATH,index=False)

print("✅ 5000 unlabeled samples generated")
print("Location:",OUTPUT_DIR)