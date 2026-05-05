import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class CNNFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)

        # Remove classification head
        self.model.fc = Identity()
        self.model.eval()
        self.model.to(self.device)

        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding.cpu().numpy().flatten()