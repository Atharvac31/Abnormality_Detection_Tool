import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNNMLP(nn.Module):

    def __init__(self, feature_dim, signal_length=512):

        super().__init__()

        # ----------------------
        # CNN branch (signal)
        # ----------------------

        self.cnn = nn.Sequential(

            nn.Conv1d(1,16,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16,32,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )

        self.cnn_fc = nn.Linear(64*16,64)


        # ----------------------
        # Feature branch
        # ----------------------

        self.feature_net = nn.Sequential(

            nn.Linear(feature_dim,64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64,32),
            nn.ReLU()
        )


        # ----------------------
        # Fusion layer
        # ----------------------

        self.classifier = nn.Sequential(

            nn.Linear(64 + 32,64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,2)
        )


    def forward(self, signal, features):

        # CNN branch

        x1 = self.cnn(signal)
        x1 = x1.view(x1.size(0),-1)
        x1 = self.cnn_fc(x1)


        # Feature branch

        x2 = self.feature_net(features)


        # Combine

        x = torch.cat([x1,x2],dim=1)

        out = self.classifier(x)

        return out