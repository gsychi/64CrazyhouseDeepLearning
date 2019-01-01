from __future__ import division
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ChessEnvironment import ChessEnvironment
import os

networkName = "Checkpoint 9 Weights"
model = torch.load("New Networks/18011810-ckpt9-POLICY.pt")
init_layer = model.conv1.weight.data.numpy().reshape((225, 3, 3))


# WE HAVE 20 BLOCKS! oops.
blocks = []
blocks.append(model.layer1[0].conv1.weight.data.numpy())
blocks.append(model.layer1[0].conv2.weight.data.numpy())
blocks.append(model.layer1[1].conv1.weight.data.numpy())
blocks.append(model.layer1[1].conv2.weight.data.numpy())
blocks.append(model.layer2[0].conv1.weight.data.numpy())
blocks.append(model.layer2[0].conv2.weight.data.numpy())
blocks.append(model.layer2[1].conv1.weight.data.numpy())
blocks.append(model.layer2[1].conv2.weight.data.numpy())
blocks.append(model.layer3[0].conv1.weight.data.numpy())
blocks.append(model.layer3[0].conv2.weight.data.numpy())
blocks.append(model.layer3[1].conv1.weight.data.numpy())
blocks.append(model.layer3[1].conv2.weight.data.numpy())
blocks.append(model.layer4[0].conv1.weight.data.numpy())
blocks.append(model.layer4[0].conv2.weight.data.numpy())
blocks.append(model.layer4[1].conv1.weight.data.numpy())
blocks.append(model.layer4[1].conv2.weight.data.numpy())
blocks.append(model.layer4[2].conv1.weight.data.numpy())
blocks.append(model.layer4[2].conv2.weight.data.numpy())
blocks.append(model.layer4[3].conv1.weight.data.numpy())
blocks.append(model.layer4[3].conv2.weight.data.numpy())


# PRINT FIRST LAYER OF CONVOLUTIONS
plt.figure(figsize=(7, 7))
for idx, filt in enumerate(init_layer):
    #print(filt[0, :, :])
    plt.subplot(15, 15, idx + 1)
    plt.imshow(filt[:, :], cmap="gray")
    plt.axis('off')
#plt.show()
saveDirec = 'Visualization of Network/' + networkName+'/Initial Conv Layer'
if not os.path.exists(saveDirec):
    os.makedirs(saveDirec)
plt.savefig(saveDirec+'/Kernels in Initial Convolutional Layer')
plt.close()


# PRINT REST OF BLOCKS
for h in range(len(blocks)):
    for i in range(15):
        plt.figure(figsize=(10, 10))
        for idx, filt in enumerate(blocks[h]):
            plt.subplot(8, 16, idx + 1)
            plt.imshow(filt[i, :, :], cmap="gray")
            title = "Kernels in Block " + str(int(h+1)) + ", Part " + str(int(i+1))
            plt.gcf().canvas.set_window_title(title)
            plt.axis('off')
        saveFolder = 'Visualization of Network/'+networkName+'/Block ' + str(int(h+1))
        saveDirec = 'Visualization of Network/'+networkName+'/Block ' + str(int(h+1)) + '/' + title
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        plt.savefig(saveDirec)
        plt.close()


board = ChessEnvironment()
representation = board.boardToState()

"""
# PRINT BOARD AND ITS REPRESENTATION
plt.figure(figsize=(4, 4))
for idx, filt in enumerate(representation[0]):
    #print(filt[0, :, :])
    plt.subplot(5, 3, idx + 1)
    plt.imshow(filt[:, :], cmap="gray")
    plt.axis('off')
plt.show()
"""


