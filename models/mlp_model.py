import torch
import torch.nn as nn


class MLPClassifier(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
)

    def forward(self, x):

        return self.model(x)