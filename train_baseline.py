from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

from dataset_loader import GraphDataset
from models.mlp_model import MLPClassifier
from feature_utils import FeatureScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ============================================================
# LOAD DATASET (IMAGE → FEATURES)
# ============================================================

dataset = GraphDataset("synthetic_dataset_v2/labels.csv")

X = []
y = []

for features, label in dataset:
    X.append(features.numpy())
    y.append(label.item())

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)   # should be (N, 17)

# ============================================================
# TRAIN / VAL SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# NORMALIZATION (VERY IMPORTANT)
# ============================================================

scaler = FeatureScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 🔥 SAVE SCALER (CRITICAL)
joblib.dump(scaler, "scaler.pkl")

# ============================================================
# CONVERT TO TENSORS
# ============================================================

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()

X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).long()

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ============================================================
# MODEL
# ============================================================

input_dim = X_train.shape[1]
print("Feature dimension:", input_dim)  # MUST be 17

model = MLPClassifier(input_dim)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# TRAINING LOOP
# ============================================================

epochs = 50

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # -------------------------
    # VALIDATION
    # -------------------------
# -------------------------
# VALIDATION (ADVANCED)
# -------------------------
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:

            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # for ROC (binary)

# Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()

# Metrics
    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {total_loss:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# ROC-AUC (only if binary)
    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"ROC-AUC: {auc:.3f}")
    except:
        pass

# Confidence analysis (🔥 useful for admin view)
confidences = np.max(np.vstack([1-all_probs, all_probs]).T, axis=1)
wrong_mask = all_preds != all_labels

if np.any(wrong_mask):
    print(f"Avg confidence (wrong preds): {confidences[wrong_mask].mean():.3f}")
print(f"Avg confidence (overall): {confidences.mean():.3f}")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(model.state_dict(), "baseline_mlp.pt")

print("✅ Training complete")
print("✅ Model saved: baseline_mlp.pt")
print("✅ Scaler saved: scaler.pkl")