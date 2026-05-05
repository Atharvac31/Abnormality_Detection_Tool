# import pandas as pd
# import torch
# from sklearn.metrics import accuracy_score
# from dataset_loader import GraphDataset
# from models.mlp_model import MLPClassifier

# # Load dataset
# dataset = GraphDataset("synthetic_dataset/labels.csv")

# X = dataset.X
# y = dataset.y

# # Load trained baseline model
# model = MLPClassifier(X.shape[1])
# model.load_state_dict(torch.load("baseline_mlp.pt"))
# model.eval()

# # Baseline predictions
# with torch.no_grad():
#     preds = model(X)
#     preds = torch.argmax(preds, dim=1)

# # Simulated LF prediction using Rules_Triggered
# df = pd.read_csv("synthetic_dataset/labels.csv")

# lf_preds = (df["Rules_Triggered"] > 1).astype(int)

# # Agreement test
# agreement = accuracy_score(preds.numpy(), lf_preds)

# print("\nWeak Supervision Diagnostic")
# print("--------------------------------")
# print("Baseline vs Rule Agreement:", agreement)

import numpy as np
from lf_features import build_lf_matrices
import pandas as pd

df = pd.read_csv("unlabeled_vibration_dataset/data.csv")

L, C, P = build_lf_matrices(df)

coverage = (L != -1).mean()
conflict = ((L == 1).sum(axis=1) > 0) & ((L == 0).sum(axis=1) > 0)

print("LF conflict:", conflict.mean())
print("LF coverage:", coverage)