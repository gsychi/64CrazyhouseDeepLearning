import numpy as np
from ChessEnvironment import ChessEnvironment
import torch
import copy
import ActionToArray
from DoubleHeadDataset import DoubleHeadDataset
import ChessResNet
import time
import math
import threading

def moveValueEvaluation(move, board, network):

    # import the network
    neuralNet = network

    tempBoard = copy.deepcopy(board)

    # import the game board
    evalBoard = ChessEnvironment()
    evalBoard.arrayBoard = tempBoard.arrayBoard
    evalBoard.board = tempBoard.board
    evalBoard.plies = tempBoard.plies
    evalBoard.whiteCaptivePieces = tempBoard.whiteCaptivePieces
    evalBoard.blackCaptivePieces = tempBoard.blackCaptivePieces
    evalBoard.actuallyAPawn = tempBoard.actuallyAPawn
    evalBoard.updateNumpyBoards()

    # make temporary move
    evalBoard.makeMove(move)

    # evalBoard.printBoard()
    state = evalBoard.boardToState()

    nullAction = torch.from_numpy(np.zeros(1))  # this will not be used, is only a filler
    testSet = DoubleHeadDataset(state, nullAction, nullAction)
    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
    with torch.no_grad():
        for images, labels1, labels2 in generatePredic:
            neuralNet.eval()
            output = (neuralNet(images)[1].numpy())[0][0]

    # so far, output gives a winning probability from -1 to 1, 1 for white, -1 for black. We want to scale this to
    # a value between 0 and 1.
    output = (output/2) + 0.5

    # now we have an evaluation from 0 to 1. Now we have to scale this to a probability
    # for either black or white depending on who moves next.
    turn = evalBoard.plies % 2

    # if plies is divisible by 2, then black has just moved, which means that
    # our evaluation should be for black. If plies is not, then white has just moved,
    # which means that our evaluation should be for white.
    if turn == 0:
        output = 1-output

    # now, let's return our evaluation
    # print(output)
    return output

def moveValueEvaluations(legalMoves, board, network):
    evaluation = np.zeros(len(legalMoves))

    class myThread(threading.Thread):
        def __init__(self, move, board, network, index):
            threading.Thread.__init__(self)
            self.move = move
            self.board = board
            self.network = network
            self.index = index

        def run(self):

            # import the network
            neuralNet = network

            tempBoard = copy.deepcopy(self.board)

            # import the game board
            evalBoard = ChessEnvironment()
            evalBoard.arrayBoard = tempBoard.arrayBoard
            evalBoard.board = tempBoard.board
            evalBoard.plies = tempBoard.plies
            evalBoard.whiteCaptivePieces = tempBoard.whiteCaptivePieces
            evalBoard.blackCaptivePieces = tempBoard.blackCaptivePieces
            evalBoard.actuallyAPawn = tempBoard.actuallyAPawn
            evalBoard.updateNumpyBoards()

            # make temporary move
            evalBoard.makeMove(self.move)

            state = evalBoard.boardToState()

            nullAction = torch.from_numpy(np.zeros(1))  # this will not be used, is only a filler
            testSet = DoubleHeadDataset(state, nullAction, nullAction)
            generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
            with torch.no_grad():
                for images, labels1, labels2 in generatePredic:
                    neuralNet.eval()
                    output = (neuralNet(images)[1].numpy())[0][0]

            # so far, output gives a winning probability from -1 to 1, 1 for white, -1 for black. We want to scale this to
            # a value between 0 and 1.
            output = (output / 2) + 0.5

            # now we have an evaluation from 0 to 1. Now we have to scale this to a probability
            # for either black or white depending on who moves next.
            turn = evalBoard.plies % 2

            # if plies is divisible by 2, then black has just moved, which means that
            # our evaluation should be for black. If plies is not, then white has just moved,
            # which means that our evaluation should be for white.
            if turn == 0:
                output = 1 - output

            # now, let's return our evaluation
            evaluation[self.index] = output

    threads = []
    for i in range(len(legalMoves)):
        t = myThread(legalMoves[i], board, network, i)
        threads.append(t)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return evaluation

def objectivePositionEval(board, network):

    # import the network
    neuralNet = network

    tempBoard = copy.deepcopy(board)

    # import the game board
    evalBoard = ChessEnvironment()
    evalBoard.arrayBoard = tempBoard.arrayBoard
    evalBoard.board = tempBoard.board
    evalBoard.plies = tempBoard.plies
    evalBoard.whiteCaptivePieces = tempBoard.whiteCaptivePieces
    evalBoard.blackCaptivePieces = tempBoard.blackCaptivePieces
    evalBoard.actuallyAPawn = tempBoard.actuallyAPawn
    evalBoard.updateNumpyBoards()

    # evalBoard.printBoard()
    state = evalBoard.boardToState()

    nullAction = torch.from_numpy(np.zeros(1))  # this will not be used, is only a filler
    testSet = DoubleHeadDataset(state, nullAction, nullAction)
    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
    with torch.no_grad():
        for images, labels1, labels2 in generatePredic:
            neuralNet.eval()
            output = (neuralNet(images)[1].numpy())[0][0]

    # so far, output gives a winning probability from -1 to 1, 1 for white, -1 for black. We want to scale this to
    # a value between 0 and 1.
    output = (output/2) + 0.5

    turn = evalBoard.plies % 2
    output = (output*2)-1
    # now, this is a probability of white winning. we need to change this to centipawns...
    output = 290.680623072 * math.tan(1.548090806 * output)

    if turn == 1:
        output = -output

    return output

def positionEval(board, network):

    # import the network
    neuralNet = network

    tempBoard = copy.deepcopy(board)

    # import the game board
    evalBoard = ChessEnvironment()
    evalBoard.arrayBoard = tempBoard.arrayBoard
    evalBoard.board = tempBoard.board
    evalBoard.plies = tempBoard.plies
    evalBoard.whiteCaptivePieces = tempBoard.whiteCaptivePieces
    evalBoard.blackCaptivePieces = tempBoard.blackCaptivePieces
    evalBoard.actuallyAPawn = tempBoard.actuallyAPawn
    evalBoard.updateNumpyBoards()

    # evalBoard.printBoard()
    state = evalBoard.boardToState()

    nullAction = torch.from_numpy(np.zeros(1))  # this will not be used, is only a filler
    testSet = DoubleHeadDataset(state, nullAction, nullAction)
    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
    with torch.no_grad():
        for images, labels1, labels2 in generatePredic:
            neuralNet.eval()
            output = (neuralNet(images)[1].numpy())[0][0]

    # so far, output gives a winning probability from -1 to 1, 1 for white, -1 for black. We want to scale this to
    # a value between 0 and 1.
    output = (output/2) + 0.5

    # now we have an evaluation from 0 to 1. Now we have to scale this to a probability
    # for either black or white depending on who moves next.
    turn = evalBoard.plies % 2

    # if plies is not divisible by 2, then it is black to move.
    if turn == 1:
        output = 1-output

    # now, let's return our evaluation
    # print(output)
    return output

testing = False
if testing:
    output = 0.55
    output = output*2-1
    output = 290.680623072 * math.tan(1.548090806 * output)
    print(output)
