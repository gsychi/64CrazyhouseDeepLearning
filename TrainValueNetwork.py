import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessConvNet import ChessConvNet
import ChessResNet
from ValueDataset import ValueDataset
import h5py

# inputs and outputs are numpy arrays. This method of checking accuracy only works with imported games.
# if it's not imported, accuracy will never be 100%, so it will just output the trained network after 10,000 epochs.
def trainValueNetwork(boards, outputs, EPOCHS=1, BATCH_SIZE=1, LR=0.001,
                 loadDirectory='none.pt',
                 saveDirectory='network1.pt', OUTPUT_ARRAY_LEN=4504):

    outputs = torch.from_numpy(outputs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = ValueDataset(boards, outputs)  # use answers instead of actions when choosing CEL

    trainLoader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)
    # to create a prediction, create a new dataset with input of the states, and output should just be np.zeros()

    # this is a convolutional neural network
    #model = ChessConvNet(OUTPUT_ARRAY_LEN).double()

    # this is a residual network
    model = ChessResNet.ValueResNet().double()

    try:
        model = torch.load(loadDirectory)
    except:
        print("Pretrained NN model not found!")

    criterion = nn.MSELoss()  # MSELoss // PoissonNLLLoss //

    # criterion = nn.CrossEntropyLoss()
    #  use this if you want to train from argmax values. This trains faster, but seems
    #  to be less accurate.

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

                    predicted = model(images).data
                    actual = labels.data
                    predicted = predicted.numpy().flatten()
                    actual = actual.numpy().flatten()
                    print(predicted)
                    print(actual)
                    print(np.abs(predicted-actual))
                    correct = 0
                    for j in range(BATCH_SIZE):
                        if np.abs(predicted-actual)[j]<0.2:
                            correct += 1
                    print("Correct:", correct)



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

    """
    with h5py.File("Training Data/18-01Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-01ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []
    """

    with h5py.File("Training Data/18-02Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-02ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="New Networks/18011810-VALUE.pt",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-03Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-03ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="New Networks/18011810-VALUE.pt",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-04Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-04ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="New Networks/18011810-VALUE.pt",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-05Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-05ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="New Networks/18011810-VALUE.pt",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-06Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-06ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="New Networks/18011810-VALUE.pt",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-07Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-07ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="New Networks/18011810-VALUE.pt",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-08Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-08ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="New Networks/18011810-VALUE.pt",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-09Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-09ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="New Networks/18011810-VALUE.pt",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

    with h5py.File("Training Data/18-10Inputs.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))
    with h5py.File("Training Data/18-10ValueOutputs.h5", 'r') as hf:
        outputs = hf["Outputs"][:]
        print(len(outputs))
    trainValueNetwork(boards, outputs, loadDirectory="New Networks/18011810-VALUE.pt",
                 saveDirectory="New Networks/18011810-VALUE.pt", EPOCHS=1,
                 BATCH_SIZE=64, LR=0.001)

    boards = []
    outputs = []

