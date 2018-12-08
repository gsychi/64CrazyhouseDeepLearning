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
    def __init__(self, block, num_blocks, num_kernels, num_classes=4504, value_net=False):
        super(ResNet, self).__init__()
        self.in_planes = 15
        self.policy_planes = block.expansion

        self.conv1 = nn.Conv2d(15, 15, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(15)
        self.layer1 = self._make_layer(block, num_kernels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_kernels[1], num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, num_kernels[2], num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, num_kernels[3], num_blocks[3], stride=1)
        self.finalLayer = nn.Sequential(
            nn.Conv2d(num_kernels[3], self.policy_planes, kernel_size=1, stride=1, padding=0),  # 64, 1
            nn.BatchNorm2d(self.policy_planes),
            nn.ReLU()
            )
        self.linear = nn.Linear(64*self.policy_planes*block.expansion, num_classes)
        self.tanh = value_net

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
        out = self.finalLayer(out)
        out = out.view(out.size(0), -1)
        if self.tanh:
            out = F.tanh(self.linear(out))
        else:
            out = F.log_softmax(self.linear(out))
        return out


def ValueResNet():
    return ResNet(BasicBlock, [2,2,2,1], [64,64,64,64], num_classes=1, value_net=True)

def PolicyResNetMain():
    return ResNet(BasicBlock, [2,2,2,4], [128,128,128,128])

def PolicyResNetSmall():
    return ResNet(BasicBlock, [2,2,2,4], [32,32,32,32])

def test():
    net = ValueResNet()
    y = net(torch.randn(1,15,8,8))
    print(y.size())
    print(y.detach().numpy())

    net = PolicyResNetMain()
    y = net(torch.randn(1,15,8,8))
    print(y.size())
    print(y.detach().numpy())
    print(np.sum(y.detach().numpy(), axis=1))

#test()
