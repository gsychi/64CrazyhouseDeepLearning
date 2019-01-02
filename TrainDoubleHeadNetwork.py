import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessConvNet import ChessConvNet
import ChessResNet
from DoubleHeadDataset import DoubleHeadDataset
import h5py

# inputs and outputs are numpy arrays. This method of checking accuracy only works with imported games.
# if it's not imported, accuracy will never be 100%, so it will just output the trained network after 10,000 epochs.
def trainDoubleHeadNetwork(boards, policyOutputs, valueOutputs, EPOCHS=1, BATCH_SIZE=1, LR=0.001,
                 loadDirectory='none.pt',
                 saveDirectory='network1.pt'):

    policyOutputs = torch.from_numpy(policyOutputs)
    valueOutputs = torch.from_numpy(valueOutputs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = DoubleHeadDataset(boards, policyOutputs, valueOutputs)

    trainLoader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    # this is a residual network
    model = ChessResNet.ResNetDoubleHeadSmall().double()

    try:
        model = torch.load(loadDirectory)
    except:
        print("Pretrained NN model not found!")

    policyCrit = nn.PoissonNLLLoss()
    valueCrit = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # , weight_decay=0.00001)
    total_step = len(trainLoader)

    trainNotFinished = True
    for epoch in range(EPOCHS):
        if trainNotFinished:
            for i, (images, labels1, labels2) in enumerate(trainLoader):
                images = images.to(device)
                policyLabels = labels1.to(device)
                valueLabels = labels2.to(device)

                # Forward pass
                outputPolicy, outputValue = model(images)

                policyLoss = policyCrit(outputPolicy, policyLabels)*4000
                valueLoss = valueCrit(outputValue, valueLabels)
                totalLoss = policyLoss + valueLoss

                # Backward and optimize
                optimizer.zero_grad()
                totalLoss.backward()
                optimizer.step()

                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Policy Loss: {:.4f}, Value Loss: {:.4f}'
                          .format(epoch + 1, EPOCHS, i + 1, total_step, policyLoss.item()/4, valueLoss.item()))
                if (i + 1) % 100 == 0:
                    # find predicted labels
                    values = np.exp((model(images)[0].data.detach().numpy()))
                    print("MAX:", np.amax(np.amax(values, axis=1)))
                    print("MIN:", np.amin(np.amin(values, axis=1)))

                    _, predicted = torch.max(model(images)[0].data, 1)
                    predicted = predicted.numpy()
                    print(predicted)

                    _, actual = torch.max(labels1.data, 1)  # for poisson nll loss
                    actual = actual.numpy()

                    print(actual)

                    print("Correct:", (predicted == actual).sum())
                if (i + 1) % 200 == 0:
                    torch.save(model, saveDirectory)

    print("Updated!")
    torch.save(model, saveDirectory)

train = True
if train:
    with h5py.File("Training Data/17-12Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/17-12PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/17-12ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="",
                 saveDirectory="New Networks/1712-smallnet.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []
