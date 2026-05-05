import numpy as np
import pandas as pd
from subset_selection import spear_ss_selection
from models.cage_aggregator import SimpleCAGE
from lf_features import build_lf_matrices
from feature_utils import FeatureScaler


# --------------------------------------------------
# 1. Load Unlabeled Dataset
# --------------------------------------------------

unlabeled_df = pd.read_csv("unlabeled_vibration_dataset/data.csv")

# separate features
features_df = unlabeled_df.drop(columns=["image_path"], errors="ignore")
X = features_df.values


# --------------------------------------------------
# 2. Feature Scaling
# --------------------------------------------------

scaler = FeatureScaler()
X = scaler.fit_transform(X)


# --------------------------------------------------
# 3. Build LF Matrices
# --------------------------------------------------

L, C, P = build_lf_matrices(features_df)


# --------------------------------------------------
# 4. CAGE Aggregation
# --------------------------------------------------

cage = SimpleCAGE(lf_acc_prior=0.75)
pseudo_labels = cage.predict(L, C)

# convert to probability (simple version)
pseudo_probs = pseudo_labels.astype(float)


# --------------------------------------------------
# 5. Run SPEAR-SS
# --------------------------------------------------

selected_indices = spear_ss_selection(
    features=X,
    probs=pseudo_probs,
    budget=100,
    filter_ratio=0.3,
    mode="supervised"
)

print(f"Selected {len(selected_indices)} samples")


# --------------------------------------------------
# 6. Save to CSV (FOR HUMAN LABELING)
# --------------------------------------------------

selected_df = unlabeled_df.iloc[selected_indices].copy()

selected_df["pseudo_prob"] = pseudo_probs[selected_indices]
selected_df["human_label"] = ""   # <-- you will fill this

selected_df.to_csv("selected_for_labeling.csv", index=False)

print("Saved to selected_for_labeling.csv")