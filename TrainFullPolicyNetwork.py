import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessConvNet import ChessConvNet
import ChessResNet
from PolicyDataset import PolicyDataset
from FullPolicyDataset import FullPolicyDataset
import h5py

# inputs and outputs are numpy arrays. This method of checking accuracy only works with imported games.
# if it's not imported, accuracy will never be 100%, so it will just output the trained network after 10,000 epochs.
def trainFullPolicyNetwork(EPOCHS=1, BATCH_SIZE=1, LR=0.001,
                 loadDirectory='none.pt',
                 saveDirectory='network1.pt'):

    data = FullPolicyDataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainLoader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)
    # to create a prediction, create a new dataset with input of the states, and output should just be np.zeros()

    # this is a residual network
    model = ChessResNet.PolicyResNetMain().double()

    try:
        model = torch.load(loadDirectory)
    except:
        print("Pretrained NN model not found!")

    criterion = nn.NLLLoss()  # MSELoss // PoissonNLLLoss //

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # , weight_decay=0.00001)
    total_step = len(trainLoader)

    trainNotFinished = True
    for epoch in range(EPOCHS):
        if trainNotFinished:
            for i, (images, labels) in enumerate(trainLoader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputMoves = model(images)

                loss = criterion(outputMoves, labels)

                if (i + 1) % 150 == 0:
                    # find predicted labels
                    values = np.exp((model(images).data.detach().numpy()))
                    print("MAX:", np.amax(np.amax(values, axis=1)))
                    print("MIN:", np.amin(np.amin(values, axis=1)))

                    _, predicted = torch.max(model(images).data, 1)
                    predicted = predicted.numpy()
                    print(predicted)

                    actual = labels.data
                    actual = actual.numpy()

                    print(actual)

                    print("Correct:", (predicted == actual).sum())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'
                          .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))
                if (i + 1) % 200 == 0:
                    torch.save(model, saveDirectory)

    print("Updated!")
    torch.save(model, saveDirectory)


# PROS: WILL TRAIN NETWORK WITHOUT USING ANY RAM
# CONS: IS VERY SLOW
trainFullPolicyNetwork(loadDirectory="",
        saveDirectory="New Networks/18011810-FULL-POLICY.pt", EPOCHS=1,
        BATCH_SIZE=1, LR=0.001)

