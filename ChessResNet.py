import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_kernels, p_planes=2, v_planes=2):
        super(ResNet, self).__init__()
        self.in_planes = 15
        self.policy_planes = p_planes
        self.value_planes = v_planes

        self.conv1 = nn.Conv2d(15, 15, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(15)
        self.layer1 = self._make_layer(block, num_kernels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_kernels[1], num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, num_kernels[2], num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, num_kernels[3], num_blocks[3], stride=1)

        # policy head
        self.finalLayerPolicy = nn.Sequential(
            nn.Conv2d(num_kernels[3], self.policy_planes, kernel_size=1, stride=1, padding=0),  # 64, 1
            nn.BatchNorm2d(self.policy_planes),
            nn.PReLU()
            )
        self.policyLinear = nn.Linear(64*self.policy_planes*block.expansion, 2308)

        # value head
        self.finalLayerValue = nn.Sequential(
            nn.Conv2d(num_kernels[3], self.value_planes, kernel_size=1, stride=1, padding=0),  # 64, 1
            nn.BatchNorm2d(self.value_planes),
            nn.PReLU()
            )
        self.valueLinear = nn.Linear(64*self.value_planes*block.expansion, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # POLICY OUTPUT
        out1 = self.finalLayerPolicy(out)
        out1 = out1.view(out1.size(0), -1)
        out1 = F.log_softmax(self.policyLinear(out1))

        # VALUE OUTPUT
        out2 = self.finalLayerValue(out)
        out2 = out2.view(out2.size(0), -1)
        out2 = torch.tanh(self.valueLinear(out2))
        return out1, out2

def ResNetDoubleHead():
    return ResNet(BasicBlock, [3,3,3,3], [256,256,256,256], p_planes=16, v_planes=8)

def ResNetDoubleHeadSmall():
    return ResNet(BasicBlock, [1,1,1,2], [32,32,32,32], p_planes=1, v_planes=4)

