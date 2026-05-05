import pandas as pd
import torch
from torch.utils.data import Dataset

class UnlabeledDataset(Dataset):

    def __init__(self, csv_path):

        df = pd.read_csv(csv_path)

        df = df.drop(columns=["image_path", "Rules_Triggered"], errors="ignore")

        self.X = df.values

        self.X = torch.tensor(self.X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx]