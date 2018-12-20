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
def trainPolicyNetwork(boards, outputs, EPOCHS=2, BATCH_SIZE=1, LR=0.001,
                 loadDirectory='none.pt',
                 saveDirectory='network1.pt', OUTPUT_ARRAY_LEN=4504):

    outputs = torch.from_numpy(outputs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = PolicyDataset(boards, outputs)  # use answers instead of actions when choosing CEL

    trainLoader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)
    # to create a prediction, create a new dataset with input of the states, and output should just be np.zeros()

    # this is a convolutional neural network
    #model = ChessConvNet(OUTPUT_ARRAY_LEN).double()

    # this is a residual network
    model = ChessResNet.PolicyResNetSmall().double()

    try:
        model = torch.load(loadDirectory)
    except:
        print("Pretrained NN model not found!")

    criterion = nn.PoissonNLLLoss()  # MSELoss // PoissonNLLLoss //


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

                if (i + 1) % 100 == 0:
                    # find predicted labels
                    values = np.exp((model(images).data.detach().numpy()))
                    print("MAX:", np.amax(np.amax(values, axis=1)))
                    print("MIN:", np.amin(np.amin(values, axis=1)))

                    _, predicted = torch.max(model(images).data, 1)
                    predicted = predicted.numpy()
                    print(predicted)

                    _, actual = torch.max(labels.data, 1)  # for poisson nll loss
                    # actual = labels.data
                    actual = actual.numpy()

                    print(actual)

                    print("Correct:", (predicted == actual).sum())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))
                if (i + 1) % 200 == 0:
                    torch.save(model, saveDirectory)

    print("Updated!")
    torch.save(model, saveDirectory)

train = True
if train:

    with h5py.File("Training Data/18-01Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-01PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-02Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-02PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-03Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-03PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-04Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-04PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-05Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-05PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-06Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-06PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-07Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-07PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-08Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-08PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-09Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-09PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-10Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-10PolicyOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainPolicyNetwork(boards, outputs, loadDirectory="New Networks/18011810-POLICY.pt",
                 saveDirectory="New Networks/18011810-POLICY.pt", EPOCHS=2,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

