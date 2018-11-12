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
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 1, 32
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32, 32
            nn.BatchNorm2d(32),  # 32
            nn.ReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # 32, 1
            nn.BatchNorm2d(1),
            nn.ReLU()
            )
        self.fc = nn.Linear(904, num_classes)

        # if dimensionality issue persists change fc into fc1 and fc2
        # fc1 = 904 x n, fc2 = n x 4504
        # less synapses if n <= 747
        # when n = 4432, 2.419 mil synapses
        # when n = 224, 1.2096 mil synapses
        # when n = 132, 604,3200 synapses.
        # self.fc = 4.036 mil synapses

        #self.fc1 = nn.Linear(904, 226)
        #self.fc2 = nn.Linear(226, 4504)

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
