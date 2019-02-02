import torch
import ChessResNet
import torchvision
from torch import nn
import h5py

# Specify the model
model = ChessResNet.ResNetDoubleHead()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Check number of trainable parameters
count = count_parameters(model)
print(count)
