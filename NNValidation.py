import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessConvNet import ChessConvNet
from PolicyDataset import PolicyDataset
from DoubleHeadDataset import DoubleHeadTrainingDataset
import ChessResNet
import h5py

# inputs and outputs are numpy arrays. This method of checking accuracy only works with imported games.
# if it's not imported, accuracy will never be 100%, so it will just output the trained network after 10,000 epochs.
def validateNetwork(loadDirectory):

    with h5py.File('Training Data/StockfishOutputs.h5', 'r') as hf:
        actions = hf["Policy Outputs"][0:1000000]
        print(len(actions))
    with h5py.File('Training Data/StockfishInputs[binaryConverted].h5', 'r') as hf:
        inputs = hf["Inputs"][0:1000000]
        print(len(inputs))
    actions = torch.from_numpy(actions)
    data = DoubleHeadTrainingDataset(inputs, actions, actions)

    testLoader = torch.utils.data.DataLoader(dataset=data, batch_size=16, shuffle=False)

    try:
        network = torch.load(loadDirectory)
        model = ChessResNet.ResNetDoubleHead().double()
        model.load_state_dict(network)
        model.eval()
    except:
        print("Pretrained NN model not found!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # eval mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, irrelevant in testLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = torch.exp(model(images)[0])
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            _, labels = torch.max(labels.data, 1)
            correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on', total, 'test positions: {:.4f} %'.format(100 * correct / total))

validate = True
if validate:
    validateNetwork("New Networks/(MCTS)(8X256|8|8)(GPU)64fish.pt")
