import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessConvNet import ChessConvNet
from PolicyDataset import PolicyDataset
import ChessResNet
import h5py

# inputs and outputs are numpy arrays. This method of checking accuracy only works with imported games.
# if it's not imported, accuracy will never be 100%, so it will just output the trained network after 10,000 epochs.
def validateNetwork(loadDirectory):

    with h5py.File('Training Data/18-11PolicyOutputs.h5', 'r') as hf:
        actions = hf["Outputs"][0:100000]
        print(len(actions))
    with h5py.File('Training Data/18-11Inputs.h5', 'r') as hf:
        inputs = hf["Inputs"][0:100000]
        print(len(inputs))
    actions = torch.from_numpy(actions)
    data = PolicyDataset(inputs, actions)

    testLoader = torch.utils.data.DataLoader(dataset=data, batch_size=128, shuffle=False)
    # to create a prediction, create a new dataset with input of the states, and output should just be np.zeros()

    try:
        checkpoint = torch.load(loadDirectory)
        model = ChessResNet.ResNetDoubleHeadSmall().double()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except:
        print("Pretrained NN model not found!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # eval mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = torch.exp(model(images)[0])
            #print(np.amax(outputs.numpy()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            _, labels = torch.max(labels.data, 1)  # for poisson nll loss
            # labels = labels.data
            correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on', total, 'test positions: {:.4f} %'.format(100 * correct / total))

validate = True
if validate:
    validateNetwork("New Networks/smallnet.pt")
