import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

from dataset_loader import GraphDataset
from models.mlp_model import MLPClassifier
from feature_utils import FeatureScaler


# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# HYPERPARAMETERS
# ======================================================
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
lambda_pseudo = 0.5

# ======================================================
# LOAD LABELED DATA (IMAGE → FEATURES)
# ======================================================
dataset = GraphDataset("synthetic_dataset_v2/labels.csv")

X, y = [], []

for features, label in dataset:
    X.append(features.numpy())
    y.append(label.item())

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)

# ======================================================
# TRAIN / VAL SPLIT
# ======================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================================
# SCALING
# ======================================================
scaler = FeatureScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# SAVE SCALER
joblib.dump(scaler, "scaler.pkl")

# ======================================================
# TO TENSOR
# ======================================================
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE
)

# ======================================================
# MODEL
# ======================================================
input_dim = X_train.shape[1]
print("Feature dim:", input_dim)

model = MLPClassifier(input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================================================
# SIMPLE PSEUDO LABELING (SELF-TRAINING)
# ======================================================
def generate_pseudo_labels(model, X_data, threshold=0.9):

    model.eval()
    pseudo_X = []
    pseudo_y = []

    with torch.no_grad():
        outputs = model(X_data)
        probs = torch.softmax(outputs, dim=1)

        conf, preds = torch.max(probs, dim=1)

        mask = conf > threshold

        pseudo_X = X_data[mask]
        pseudo_y = preds[mask]

    return pseudo_X, pseudo_y


# ======================================================
# TRAINING LOOP
# ======================================================
best_acc = 0
patience = 20
patience_counter = 0

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    # ----------------------------------
    # Generate pseudo labels every epoch
    # ----------------------------------
    pseudo_X, pseudo_y = generate_pseudo_labels(model, X_train)

    for X_batch, y_batch in train_loader:

        # ----------------------
        # Supervised loss
        # ----------------------
        preds = model(X_batch)
        supervised_loss = F.cross_entropy(preds, y_batch)

        # ----------------------
        # Pseudo loss (if exists)
        # ----------------------
        if len(pseudo_X) > 0:

            pseudo_preds = model(pseudo_X)
            pseudo_loss = F.cross_entropy(pseudo_preds, pseudo_y)

            loss = supervised_loss + lambda_pseudo * pseudo_loss

        else:
            loss = supervised_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

# ==================================================
# VALIDATION (ADVANCED)
# ==================================================
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
            all_probs.extend(probs[:, 1].cpu().numpy())

# Convert
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()

    print(f"\nEpoch {epoch+1}")
    print(f"Loss {total_loss:.3f}")
    print(f"Accuracy {accuracy:.3f}")

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"ROC-AUC: {auc:.3f}")
    except:
        pass

# Confidence analysis
    confidences = np.max(np.vstack([1-all_probs, all_probs]).T, axis=1)
    wrong_mask = all_preds != all_labels

    if np.any(wrong_mask):
        print(f"Avg confidence (wrong preds): {confidences[wrong_mask].mean():.3f}")
    print(f"Avg confidence (overall): {confidences.mean():.3f}")

    # ==================================================
    # EARLY STOPPING
    # ==================================================
    if accuracy > best_acc:
        best_acc = accuracy
        patience_counter = 0
        torch.save(model.state_dict(), "joint_learning_mlp.pt")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("\nEarly stopping triggered")
        break

print("\n✅ Model saved as joint_learning_mlp.pt")