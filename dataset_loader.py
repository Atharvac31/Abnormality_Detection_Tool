import pandas as pd
import torch
from torch.utils.data import Dataset


# ------------------------------------------------
# Feature Dataset (Used for MLP / baseline model)
# ------------------------------------------------

class GraphDataset(Dataset):

    def __init__(self, csv_path):

        df = pd.read_csv(csv_path)

        # Store image paths separately
        self.image_paths = df["image_path"].values

        # Labels
        self.labels = df["Abnormality"].values

        # Feature columns
        feature_df = df.drop(columns=["image_path", "Abnormality", "Rules_Triggered"], errors="ignore")

        self.features = feature_df.values


    def __len__(self):

        return len(self.features)


    def __getitem__(self, idx):

        X = torch.tensor(self.features[idx], dtype=torch.float32)

        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return X, y


# ------------------------------------------------
# Hybrid Dataset (Signal + Features)
# ------------------------------------------------

class HybridDataset(Dataset):

    def __init__(self, csv_path, signal_loader):

        df = pd.read_csv(csv_path)

        self.image_paths = df["image_path"].values
        self.labels = df["Abnormality"].values

        feature_df = df.drop(columns=["image_path", "Abnormality", "Rules_Triggered"], errors="ignore")

        self.features = feature_df.values

        # function used to convert image → signal
        self.signal_loader = signal_loader


    def __len__(self):

        return len(self.features)


    def __getitem__(self, idx):

        features = torch.tensor(self.features[idx], dtype=torch.float32)

        signal = self.signal_loader(self.image_paths[idx])

        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return signal, features, label