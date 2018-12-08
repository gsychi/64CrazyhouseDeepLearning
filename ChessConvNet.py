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
            nn.Conv2d(15, 64, kernel_size=3, stride=1, padding=1),  # 1, 64
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64, 64
            nn.BatchNorm2d(64),  # 64
            nn.ReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64, 64
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64, 64
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64, 64
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64, 64
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),  # 64, 1
            nn.BatchNorm2d(1),
            nn.ReLU()
            )
        self.fc = nn.Linear(8*8, num_classes)

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

        # out = F.relu(self.fc1(out))
        # out = self.fc2(out)
        return out
