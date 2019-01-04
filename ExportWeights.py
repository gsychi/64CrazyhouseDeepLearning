"""
When we train networks via TrainDoubleHeadNetwork.py, the saved .pt file contains
information on the state dict, the training loss, and the optimizer gradients.

This, as expected, takes a lot more memory. So here, we're taking a training checkpoint
and converting it to a pytorch file with weights
"""
import torch
import ChessResNet

# Specify the model
model = ChessResNet.ResNetDoubleHead()

# Load training checkpoint
savedFile = torch.load("New Networks/smallnet.pt")
model.load_state_dict(savedFile['model_state_dict'])

# Save Weights
torch.save(model.state_dict(), "New Networks/8X256-PV.pt")

