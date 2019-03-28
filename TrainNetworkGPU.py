import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
import ChessResNet
from DoubleHeadDataset import DoubleHeadTrainingDataset
import h5py


# inputs and outputs are numpy arrays. This method of checking accuracy only works with imported games.
# if it's not imported, accuracy will never be 100%, so it will just output the trained network after 10,000 epochs.
def trainGPUNetwork(boards, policyOutputs, policyMag, valueOutputs, EPOCHS=1, BATCH_SIZE=1, LR=0.001,
                           loadDirectory='none.pt', saveDirectory='network1.pt'):

    policyLossHistory = []
    valueLossHistory = []

    policyOutputs = torch.from_numpy(policyOutputs).double()
    valueOutputs = torch.from_numpy(valueOutputs).double()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = DoubleHeadTrainingDataset(boards, policyOutputs, policyMag,valueOutputs)

    trainLoader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    # this is a residual network
    model = ChessResNet.ResNetDoubleHead().double().cuda()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    try:
        checkpoint = torch.load(loadDirectory)
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

                policyLossHistory.append(policyLoss.detach().cpu().numpy())
                valueLossHistory.append(valueLoss.detach().cpu().numpy())

                # Backward and optimize
                optimizer.zero_grad()
                totalLoss.backward()
                optimizer.step()

                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Policy Loss: {:.4f}, Value Loss: {:.4f}'
                          .format(epoch + 1, EPOCHS, i + 1, total_step, policyLoss.item() / 4, valueLoss.item()))
                if (i + 1) % 200 == 0:
                    # find predicted labels
                    values = np.exp((model(images)[0].data.detach().cpu().numpy()))
                    print("MAX:", np.amax(np.amax(values, axis=1)))
                    print("MIN:", np.amin(np.amin(values, axis=1)))

                    _, predicted = torch.max(model(images)[0].data, 1)
                    predicted = predicted.cpu().numpy()
                    print(predicted)

                    _, actual = torch.max(labels1.data, 1)  # for poisson nll loss
                    actual = actual.numpy()

                    print(actual)

                    print("Correct:", (predicted == actual).sum())
                if (i + 1) % 400 == 0:
                    # Save Model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': totalLoss,
                    }, saveDirectory)

                    # Save Loss History
                    outF = open("Network Logs/policyLossHistory.txt", "w")
                    for k in range(len(policyLossHistory)):
                        # write line to output file
                        outF.write(str(policyLossHistory[k]))
                        outF.write("\n")
                    outF.close()
                    outF = open("Network Logs/valueLossHistory.txt", "w")
                    for l in range(len(valueLossHistory)):
                        # write line to output file
                        outF.write(str(valueLossHistory[l]))
                        outF.write("\n")
                    outF.close()


    print("Updated!")
    # Save Model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': totalLoss,
    }, saveDirectory)

    # Save Loss History
    outF = open("Network Logs/policyLossHistory.txt", "w")
    for m in range(len(policyLossHistory)):
        # write line to output file
        outF.write(str(policyLossHistory[m]))
        outF.write("\n")
    outF.close()
    outF = open("Network Logs/valueLossHistory.txt", "w")
    for n in range(len(valueLossHistory)):
        # write line to output file
        outF.write(str(valueLossHistory[n]))
        outF.write("\n")
    outF.close()



train = True
if train:

    with h5py.File("Training Data/StockfishInputs3[binaryConverted].h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/StockfishOutputs3.h5", 'r') as hf:
        policy = hf["Policy Outputs"][:]
        policyMag = hf["Policy Magnitude Outputs"][:]
        value = hf["Value Outputs"][:]
        print(len(value))
    trainGPUNetwork(boards, policy, policyMag, value, loadDirectory="New Networks/[12x256_16_8]64fish.pt",
                           saveDirectory="New Networks/[12x256_16_8]64fish.pt", EPOCHS=2,
                           BATCH_SIZE=128, LR=0.001)

