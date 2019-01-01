import numpy as np
from ChessEnvironment import ChessEnvironment
import torch
from MyDataset import MyDataset
import copy
import ActionToArray

def moveValueEvaluation(move, board, network):

    # import the network
    valueNet = network

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

    nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
    testSet = MyDataset(state, nullAction)
    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
    with torch.no_grad():
        for images, labels in generatePredic:
            valueNet.eval()
            output = (valueNet(images).numpy())[0][0]

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
    evaluations = np.zeros(len(legalMoves))
    for i in range(len(evaluations)):
        evaluations[i] = moveValueEvaluation(legalMoves[i], board, network)
    return evaluations

def objectivePositionEval(board, network):

    # import the network
    valueNet = network

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

    nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
    testSet = MyDataset(state, nullAction)
    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
    with torch.no_grad():
        for images, labels in generatePredic:
            valueNet.eval()
            output = (valueNet(images).numpy())[0][0]

    # so far, output gives a winning probability from -1 to 1, 1 for white, -1 for black. We want to scale this to
    # a value between 0 and 1.
    output = (output/2) + 0.5

    turn = evalBoard.plies % 2

    # now, this is a probability of white winning. we need to change this to centipawns...
    output = min(1, 4*np.arctanh(output-0.500001))

    if turn == 1:
        output = -output

    return output

def positionEval(board, network):

    # import the network
    valueNet = network

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

    nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
    testSet = MyDataset(state, nullAction)
    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
    with torch.no_grad():
        for images, labels in generatePredic:
            valueNet.eval()
            output = (valueNet(images).numpy())[0][0]

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
    hi = ChessEnvironment()

    hi.printBoard()
    moves = ActionToArray.legalMovesForState(hi.arrayBoard, hi.board)
    print(moves)
    network = torch.load("18011802-VALUE.pt")
    network.eval()
    evaluations = moveValueEvaluations(moves, hi, network)
    print(evaluations)
    eval = positionEval(hi, network)
    print(eval)



