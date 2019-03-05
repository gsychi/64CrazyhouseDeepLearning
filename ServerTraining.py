import torch.utils.data as data_utils
import chess.variant
import torch
import numpy as np
import torch.nn as nn
from ChessEnvironment import ChessEnvironment
from ChessConvNet import ChessConvNet
from MyDataset import MyDataset
import ActionToArray
import pathlib


# inputs and outputs are numpy arrays. This method of checking accuracy only works with imported games.
# if it's not imported, accuracy will never be 100%, so it will just output the trained network after 10,000 epochs.
def trainNetwork(states, outputMoves, EPOCHS=10000, BATCH_SIZE=1000, LR=0.001, loadDirectory='none.pt',
                 saveDirectory='network1.pt', OUTPUT_ARRAY_LEN=4504, THRESHOLD_FOR_SAVE=100, updateInterval=1):
    states = torch.from_numpy(states)
    outputMoves = torch.from_numpy(outputMoves)
    answers = (np.argmax(outputMoves, axis=1)).long()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    boards, actions = states, outputMoves

    data = MyDataset(boards, answers)  # use answers instead of actions when choosing CEL

    trainLoader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=data, batch_size=len(boards), shuffle=False)
    # to create a prediction, create a new dataset with input of the states, and output should just be np.zeros()

    # TRAINING!

    # this is a convolutional neural network
    model = ChessConvNet(OUTPUT_ARRAY_LEN).double()

    # this is a residual network
    # model = ChessResNet.ResNet18().double()

    try:
        model = torch.load(loadDirectory)
    except:
        print("Pretrained NN model not found!")

    criterion = nn.PoissonNLLLoss()  # use this if you want to train from pick up square as well

    criterion = nn.CrossEntropyLoss()
    #  use this if you want to train from argmax values. This trains faster,
    # but only trains one value as best move instead of weighted probability.

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_step = len(trainLoader)

    bestAccuracy = 0
    trainNotFinished = True
    for epoch in range(EPOCHS):
        if trainNotFinished:
            for i, (images, labels) in enumerate(trainLoader):
                images = images.to(device)
                labels = labels.to(device)
                if epoch >= 150:
                    LR = 0.01
                elif epoch >= 250:
                    LR = 0.001

                # Forward pass
                outputMoves = model(images)
                loss = criterion(outputMoves, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))
            torch.save(model, saveDirectory)
            if epoch % updateInterval == 999:
                # Test the model
                model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
                answers = np.argmax(actions.numpy(), axis=1)
                with torch.no_grad():
                    for images, labels in testLoader:
                        images = images.to(device)
                        # labels = labels.to(device)
                        outputMoves = model(images)
                        _, predicted = torch.max(outputMoves.data, 1)

                        # print expectations vs reality
                        print("MAX", np.amax(outputMoves.numpy()))
                        print("MIN", np.amin(outputMoves.numpy()))

                        print(predicted.numpy())
                        print(answers)

                        correct = (predicted.numpy() == answers).sum()
                        acc = 100 * (correct / len(answers))
                        print("argmax prediction: ", acc, "% correct.")

                        if epoch == 1:
                            bestAccuracy = acc
                        else:
                            if acc > bestAccuracy:
                                bestAccuracy = acc
                                torch.save(model, saveDirectory)

                        if acc >= THRESHOLD_FOR_SAVE:
                            torch.save(model, saveDirectory)
                            print("Updated!")
                            trainNotFinished = False

                        print("best accuracy: ", bestAccuracy, "% correct.")

    print("Updated!")
    torch.save(model, saveDirectory)


train = True
if train:

    pgnGames = list(pathlib.Path('lichessdatabase').glob('*.pgn'))
    listOfMoves = []
    for i in range(len(pgnGames)):
        pgn = open(pgnGames[i])
        for k in range(190000):  # 190,000 assures all games are looked at.
            try:
                game = chess.pgn.read_game(pgn)
                whiteElo = int(game.headers["WhiteElo"])
                blackElo = int(game.headers["BlackElo"])
                benchmark = 2450
                if whiteElo >= benchmark and blackElo >= benchmark:
                    print(whiteElo)
                    print(blackElo)
                    board = game.board()
                    singleGame = []
                    for move in game.main_line():
                        board.push(move)
                        singleGame.append(move.uci())
                    listOfMoves.append(singleGame)
                    print(pgnGames[i])
            except:
                print("", end="")

    f = open("2018games2000.txt", "w+")
    for i in range(len(listOfMoves)):
        print(listOfMoves[i], ",")
        f.write(str(listOfMoves[i]) + ",\n")
    f.close()

    inList = []
    outList = []

    for j in range(len(listOfMoves)):
        board = ChessEnvironment()
        for i in range(len(listOfMoves[j])):
            state = board.boardToState()
            action = ActionToArray.moveArray(listOfMoves[j][i], board.arrayBoard)
            if board.board.legal_moves.count() != len(ActionToArray.legalMovesForState(board.arrayBoard, board.board)):
                print("ERROR!")

            board.makeMove(listOfMoves[j][i])
            # check if state is seen before

            seenBefore = False
            directory = -1
            for k in range(len(inList)):
                if np.sum(abs(inList[k].flatten() - state.flatten())) == 0:
                    seenBefore = True
                    directory = k
            if not seenBefore:
                inList.append(state)
                outList.append(action)
            else:
                outList[directory] += action

        print(board.board)
        board.gameResult()
        print(board.gameStatus)
        print(len(inList))
        print(len(outList))
        print(str(int(j + 1)), "out of ", len(listOfMoves), "parsed.")

    # all games are parsed, now convert list into array
    inputs = np.zeros((len(inList), 1, 32, 28))
    outputs = np.zeros((len(outList), 4504))

    for i in range(len(inList)):
        inputs[i] = inList[i][0]
        outputs[i] = outList[i][0]

    print(inputs.shape)
    print(outputs.shape)

    # normalize data so that the largest possible value of a move is 1.
    for i in range(len(outputs)):
        for j in range(4504):
            outputs[i][j] /= np.amax(outputs[i])

    # the computer does not seem to be placing pieces so let's change that.

    print("downloading done")
    print(inputs.shape)
    print(np.amax(outputs, axis=1))
    print(np.sum(outputs, axis=1))

    # if you want a random network
    # inputs = np.zeros((1, 1, 112, 8))
    # outputs = np.zeros((1, 4504))

    # Now, with this database, we start training the neural network.
    trainNetwork(inputs, outputs, loadDirectory="supervisedaL.pt", saveDirectory="supervisedSMALL.pt", EPOCHS=1000,
                 BATCH_SIZE=64, updateInterval=1, LR=0.01)  # 0.005

