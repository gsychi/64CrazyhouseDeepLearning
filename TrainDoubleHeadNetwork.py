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
    model = ChessResNet.ResNetDoubleHead().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    try:
        checkpoint = torch.load(saveDirectory)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        totalLoss = checkpoint['loss']

    except:
        print("Pretrained NN model not found!")

    policyCrit = nn.PoissonNLLLoss()
    valueCrit = nn.MSELoss()

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

                policyLoss = policyCrit(outputPolicy, policyLabels) * 4000
                valueLoss = valueCrit(outputValue, valueLabels)
                totalLoss = policyLoss + valueLoss

                # Backward and optimize
                optimizer.zero_grad()
                totalLoss.backward()
                optimizer.step()

                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Policy Loss: {:.4f}, Value Loss: {:.4f}'
                          .format(epoch + 1, EPOCHS, i + 1, total_step, policyLoss.item() / 4, valueLoss.item()))
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
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': totalLoss,
                    }, saveDirectory)

    print("Updated!")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': totalLoss,
    }, saveDirectory)


train = True
if train:

    with h5py.File("Training Data/18-02Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-02PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/18-02ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="New Networks/smallnet.pt",
                           saveDirectory="New Networks/smallnet.pt", EPOCHS=1,
                           BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-03Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-03PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/18-03ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="New Networks/smallnet.pt",
                           saveDirectory="New Networks/smallnet.pt", EPOCHS=1,
                           BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-04Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-04PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/18-04ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="New Networks/smallnet.pt",
                           saveDirectory="New Networks/smallnet.pt", EPOCHS=1,
                           BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-05Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-05PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/18-05ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="New Networks/smallnet.pt",
                           saveDirectory="New Networks/smallnet.pt", EPOCHS=1,
                           BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-06Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-06PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/18-06ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="New Networks/smallnet.pt",
                           saveDirectory="New Networks/smallnet.pt", EPOCHS=1,
                           BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-07Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-07PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/18-07ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="New Networks/smallnet.pt",
                           saveDirectory="New Networks/smallnet.pt", EPOCHS=1,
                           BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-08Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-08PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/18-08ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="New Networks/smallnet.pt",
                           saveDirectory="New Networks/smallnet.pt", EPOCHS=1,
                           BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-09Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-09PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/18-09ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="New Networks/smallnet.pt",
                           saveDirectory="New Networks/smallnet.pt", EPOCHS=1,
                           BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-10Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-10PolicyOutputs.h5", 'r') as hf:
        policy = hf["Outputs"][:]
        print(len(policy))
    with h5py.File("Training Data/18-10ValueOutputs.h5", 'r') as hf:
        value = hf["Outputs"][:]
        print(len(value))
    trainDoubleHeadNetwork(boards, policy, value, loadDirectory="New Networks/smallnet.pt",
                           saveDirectory="New Networks/smallnet.pt", EPOCHS=1,
                           BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []
