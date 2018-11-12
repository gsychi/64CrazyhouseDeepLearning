import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessConvNet import ChessConvNet
from TrainingDataset import TrainingDataset
import ChessResNet
import h5py

# inputs and outputs are numpy arrays. This method of checking accuracy only works with imported games.
# if it's not imported, accuracy will never be 100%, so it will just output the trained network after 10,000 epochs.
def validateEnsembleNetwork():

    with h5py.File('Training Data/Old Data/validationOutputs.h5', 'r') as hf:
        actions = hf["Outputs"][:]
        print(len(actions))
    with h5py.File('Training Data/Old Data/validationInputs.h5', 'r') as hf:
        inputs = hf["Inputs"][:]
        print(len(inputs))
    actions = torch.from_numpy(actions)
    data = TrainingDataset(inputs, actions)

    testLoader = torch.utils.data.DataLoader(dataset=data, batch_size=16, shuffle=False)
    # to create a prediction, create a new dataset with input of the states, and output should just be np.zeros()

    try:
        model1 = torch.load("7 Layer Old Input Models/1610to1810.pt")
        model2 = torch.load("7 Layer Old Input Models/v6-1803to1806.pt")
        model3 = torch.load("7 Layer Old Input Models/1705to1810.pt")
        model4 = torch.load("7 Layer Old Input Models/v3-1712to1803.pt")
        model5 = torch.load("7 Layer Old Input Models/v5-1706to1809.pt")
        model6 = torch.load("7 Layer Old Input Models/1611to1810.pt")
        print("6 TOTAL MODELS")

    except:
        print("Pretrained NN model not found!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1.eval()  # eval mode
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs1 = model1(images)
            outputs2 = model2(images)
            outputs3 = model3(images)
            outputs4 = model4(images)
            outputs5 = model5(images)
            outputs6 = model6(images)
            outputs = outputs1+outputs2+outputs3+outputs4+outputs5+outputs6
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            _, labels = torch.max(labels.data, 1)
            correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on', total, 'test positions: {} %'.format(100 * correct / total))

validate = True
if validate:
    validateEnsembleNetwork()
    print("ensemble net")




