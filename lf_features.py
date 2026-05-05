import numpy as np


# ------------------------------------------------
# Feature Engineering
# ------------------------------------------------

def compute_derived_features(df):

    df = df.copy()
    eps = 1e-6

    # Existing derived vibration features
    df["Energy_Ratio"] = df["Variance"] / (df["RMS"] + eps)

    df["Spike_Intensity"] = df["Peak_Count"] / (df["Std"] + eps)

    df["Shape_Factor"] = df["RMS"] / (np.abs(df["Mean"]) + eps)

    df["Crest_Factor"] = df["Max"] / (df["RMS"] + eps)

    df["Variability_Index"] = df["Std"] / (np.abs(df["Mean"]) + eps)

    df["Spectral_Entropy"] = -(df["Energy_Ratio"] * np.log(df["Energy_Ratio"] + eps))

    # Interaction features

    df["Shock_Energy"] = df["RMS"] * df["Max"]

    df["Impulsive_Fault_Score"] = df["Kurtosis"] * df["Peak_Count"]

    df["Instability_Interaction"] = df["Variance"] * df["Std"]

    df["Drift_Energy"] = np.abs(df["Mean"]) * df["RMS"]

    df["Spike_Energy"] = df["Peak_Count"] * df["Variance"]

    # ------------------------------------------------
    # SPC-Based Anomaly Features
    # ------------------------------------------------

    df["CLV_Score"] = np.abs(df["Mean"]) / (df["Std"] + eps)

    df["Mean_Shift_Score"] = np.abs(df["Mean"]) / (df["RMS"] + eps)

    df["Instability_Score"] = df["Variance"] / ((df["RMS"] ** 2) + eps)

    df["Spike_Cluster_Score"] = df["Peak_Count"] / (df["RMS"] + eps)

    df["Oscillation_Score"] = df["Std"] / (df["RMS"] + eps)

    return df


# ------------------------------------------------
# Compute Adaptive Thresholds
# ------------------------------------------------

def compute_thresholds(df):

    T = {}

    # anomaly thresholds
    T["variance_high"] = np.percentile(df["Variance"], 90)
    T["kurtosis_high"] = np.percentile(df["Kurtosis"], 90)
    T["spike_high"] = np.percentile(df["Peak_Count"], 90)
    T["skew_high"] = np.percentile(np.abs(df["Skewness"]), 90)
    T["freq_high"] = np.percentile(df["Dominant_Freq"], 90)

    T["energy_ratio_high"] = np.percentile(df["Energy_Ratio"], 85)
    T["spike_intensity_high"] = np.percentile(df["Spike_Intensity"], 85)
    T["crest_high"] = np.percentile(df["Crest_Factor"], 85)
    T["variability_high"] = np.percentile(df["Variability_Index"], 85)
    T["entropy_high"] = np.percentile(df["Spectral_Entropy"], 85)

    # SPC thresholds
    T["clv_high"] = np.percentile(df["CLV_Score"], 90)
    T["shift_high"] = np.percentile(df["Mean_Shift_Score"], 90)
    T["instability_high"] = np.percentile(df["Instability_Score"], 90)
    T["spike_cluster_high"] = np.percentile(df["Spike_Cluster_Score"], 90)
    T["oscillation_high"] = np.percentile(df["Oscillation_Score"], 90)

    # normal thresholds
    T["variance_low"] = np.percentile(df["Variance"], 10)
    T["spike_low"] = np.percentile(df["Peak_Count"], 10)
    T["skew_low"] = np.percentile(np.abs(df["Skewness"]), 10)
    T["plateau_low"] = np.percentile(df["Plateau_Ratio"], 20)

    return T


# ------------------------------------------------
# Abnormal Labeling Functions
# ------------------------------------------------

def lf_high_variance(x, T):
    if x["Variance"] > T["variance_high"]:
        return 1, 0.9
    return -1, 0


def lf_heavy_tail(x, T):
    if x["Kurtosis"] > T["kurtosis_high"]:
        return 1, 0.85
    return -1, 0


def lf_spike_pattern(x, T):
    if x["Peak_Count"] > T["spike_high"]:
        return 1, 0.85
    return -1, 0


def lf_skewed_signal(x, T):
    if abs(x["Skewness"]) > T["skew_high"]:
        return 1, 0.8
    return -1, 0


def lf_energy_instability(x, T):
    if x["Energy_Ratio"] > T["energy_ratio_high"]:
        return 1, 0.9
    return -1, 0


def lf_impulsive_signal(x, T):
    if x["Crest_Factor"] > T["crest_high"]:
        return 1, 0.8
    return -1, 0


# ------------------------------------------------
# SPC Labeling Functions
# ------------------------------------------------

def lf_control_limit_violation(x, T):
    if x["CLV_Score"] > T["clv_high"]:
        return 1, 0.9
    return -1, 0


def lf_mean_shift(x, T):
    if x["Mean_Shift_Score"] > T["shift_high"]:
        return 1, 0.85
    return -1, 0


def lf_high_instability(x, T):
    if x["Instability_Score"] > T["instability_high"]:
        return 1, 0.85
    return -1, 0


def lf_spike_cluster(x, T):
    if x["Spike_Cluster_Score"] > T["spike_cluster_high"]:
        return 1, 0.85
    return -1, 0


def lf_oscillation(x, T):
    if x["Oscillation_Score"] > T["oscillation_high"]:
        return 1, 0.8
    return -1, 0


# ------------------------------------------------
# Normal Labeling Functions
# ------------------------------------------------

def lf_low_variance(x, T):
    if x["Variance"] < T["variance_low"]:
        return 0, 0.75
    return -1, 0


def lf_no_spikes(x, T):
    if x["Peak_Count"] < T["spike_low"]:
        return 0, 0.7
    return -1, 0


def lf_stable_signal(x, T):
    if abs(x["Skewness"]) < T["skew_low"] and x["Kurtosis"] < 3:
        return 0, 0.75
    return -1, 0


def lf_flat_signal(x, T):
    if x["Plateau_Ratio"] < T["plateau_low"]:
        return 0, 0.7
    return -1, 0


# ------------------------------------------------
# LF List
# ------------------------------------------------

LF_LIST = [

    # statistical anomaly
    lf_high_variance,
    lf_heavy_tail,
    lf_spike_pattern,
    lf_skewed_signal,
    lf_energy_instability,
    lf_impulsive_signal,

    # SPC anomaly rules
    lf_control_limit_violation,
    lf_mean_shift,
    lf_high_instability,
    lf_spike_cluster,
    lf_oscillation,

    # normal rules
    lf_low_variance,
    lf_no_spikes,
    lf_stable_signal,
    lf_flat_signal
]


# ------------------------------------------------
# Apply LFs
# ------------------------------------------------

def apply_lfs(row, T):

    labels = []
    confidences = []

    for lf in LF_LIST:
        label, conf = lf(row, T)
        labels.append(label)
        confidences.append(conf)

    return labels, confidences


# ------------------------------------------------
# Build LF matrices
# ------------------------------------------------

def build_lf_matrices(df):

    df = compute_derived_features(df)
    T = compute_thresholds(df)

    L = []
    C = []

    for _, row in df.iterrows():

        labels, confs = apply_lfs(row, T)

        L.append(labels)
        C.append(confs)

    return np.array(L), np.array(C), T