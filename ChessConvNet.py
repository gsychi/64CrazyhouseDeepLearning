import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class ChessConvNet(nn.Module):
    def __init__(self, num_classes):
        self.numClasses = num_classes
        super(ChessConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, stride=1, padding=2),  # 1, 64
            nn.BatchNorm2d(12),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=5, stride=1, padding=2),  # 64, 12
            nn.BatchNorm2d(12),  # 12
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),  # 12, 12
            nn.BatchNorm2d(12),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),  # 12, 12
            nn.BatchNorm2d(12),
            nn.ReLU())
        self.fc = nn.Linear(896 * 12, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


"""
icga = game competitions
"""