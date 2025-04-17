# src/model.py
import torch
import torch.nn as nn
# from torchvision.models import densenet201
from torchvision.models import densenet201, DenseNet201_Weights

class DenseNet201Classifier(nn.Module):
    def __init__(self, use_pretrained=True):
        super(DenseNet201Classifier, self).__init__()
        weights = DenseNet201_Weights.DEFAULT if use_pretrained else None
        # self.model = densenet201(weights=weights)
        self.model = densenet201(weights=weights)
        num_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(num_features, 1)
        self.model.classifier = nn.Linear(num_features, 1)


    def forward(self, x):
        return self.model(x)

import torch.nn.functional as F


# 2 LAYERS

# class HistopathologicCNN(nn.Module):
#     def __init__(self):
#         super(HistopathologicCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.25)
        
#         # ✅ Updated to correct input size: 64 * 22 * 22 = 30976
#         self.fc1 = nn.Linear(30976, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # Conv1 → ReLU → Pool
#         x = self.pool(F.relu(self.conv2(x)))  # Conv2 → ReLU → Pool
#         # print("📐 Shape before flatten:", x.shape)
#         x = x.view(x.size(0), -1)             # Dynamically flatten
#         # print("📐 Flattened shape:", x.shape[1])
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x





