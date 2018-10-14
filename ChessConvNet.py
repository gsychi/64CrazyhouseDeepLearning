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
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # 1, 16
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),  # 16, 16
            nn.BatchNorm2d(32),  # 16
            nn.ReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 16, 8
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),  # 8, 4
            nn.BatchNorm2d(1),
            nn.ReLU()
            )
        self.fc = nn.Linear(896 * 1, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


"""
icga = game competitions
"""
