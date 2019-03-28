"""

Since strings can be turned into arrays, and these arrays can be turned back into the string for the move,
it is now time to create self-learning training data for the computer!

This will be done through MCTS.

At each position, there will be n number of legal moves.

Using the legalMovesFromSquare and MoveablePieces framework, we can create a list of moves for each position.
These nodes will then be updated each time a new playout is finished.
"""
import datetime

import numpy as np
import copy
import chess.variant
import chess.pgn
import chess
import time
from ChessEnvironment import ChessEnvironment
import ActionToArray
import ChessResNet
import _thread
import ChessConvNet
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from DoubleHeadDataset import DoubleHeadDataset
import ValueEvaluation
from DoubleHeadDataset import DoubleHeadTrainingDataset

# w stands for # of wins, n stands for number of times node has been visited.
# N stands for number of times parent node is visited, and c is just an exploration constant that can be tuned.
# q is the value evaluation from 0 to 1 of the neural network
# p is the probability evaluation from 0 to 1 of the neural network
# UCT Algorithm used by Alpha Zero.
def PUCT_Algorithm(w, n, c, N, q, p):
    # Provides a win rate score from 0 to 1
    selfPlayEvaluation = w/n
    selfPlayEvaluation[selfPlayEvaluation == float("Inf")] = 0
    selfPlayEvaluation[selfPlayEvaluation != selfPlayEvaluation] = 0
    winRate = (q + selfPlayEvaluation) * 0.5

    # Exploration
    exploration = (c * p * torch.sqrt(N)) / (1 + n)

    PUCT = winRate + exploration

    return PUCT

def noiseEvals(nnEvals, bounds):
    # this diversifies training data during its self-play games, in order to ensure that the computer looks at a lot of
    # different positions.
    noise = (torch.rand(len(nnEvals)) * 2 * bounds) - (bounds)
    return noise + nnEvals

class MCTSTensor():

    # We will use three lists:
    # seenStates stores the gameStates that have been seen. This is a library.
    # parentSeen stores the times that the current gameState has been seen
    # Each game state corresponds to arrays of the possible moves
    # There are 3 points information stored for each of the children
    # - win count, number of times visited, and neural network evaluation
    # This is helpful because we get to use numpy stuffs.

    def __init__(self, directory, depth):

        self.dictionary = {
            # 'string' = n position. Let this string be the FEN of the position.
        }
        self.childrenMoveNames = []  # a 2D list, each directory may be of different size, stores name of moves
        self.childrenStateSeen = []  # a 2D list, each directory contains numpy array
        self.childrenStateWin = []  # a 2D list, each directory contains numpy array
        self.childrenPolicyEval = []  # a 2D list, each directory contains numpy array
        self.childrenValueEval = []  # a 2D list, each directory contains numpy array
        self.neuralNet = ChessResNet.ResNetDoubleHead().double()

        try:
            self.neuralNet.load_state_dict(torch.load(directory))
            self.neuralNet.eval()
        except:
            print("Network not found!")

        self.nameOfNetwork = directory[0:-3]
        self.DEPTH_VALUE = depth

# This adds information into the MCTS database

    def clearInformation(self):
        self.dictionary = {
            # 'string' = n position. Let this string be the FEN of the position.
        }
        self.childrenMoveNames = []  # a 2D list, each directory may be of different size, stores name of moves
        self.childrenStateSeen = []  # a 2D list, each directory contains numpy array
        self.childrenStateWin = []  # a 2D list, each directory contains numpy array
        self.childrenPolicyEval = []  # a 2D list, each directory contains numpy array
        self.childrenValueEval = []  # a 2D list, each directory contains numpy array

    def printInformation(self):
        print(self.dictionary)
        print(self.childrenMoveNames)
        print(self.childrenStateSeen)
        print(self.childrenStateWin)
        print(self.childrenPolicyEval)
        print(self.childrenValueEval)
        print("Parent states in tree: ", len(self.childrenMoveNames))

    def printSize(self):
        print("Size:", len(self.childrenMoveNames))

    def addPositionToMCTS(self, string, legalMoves, arrayBoard, prediction):
        start = time.time()
        self.dictionary[string] = len(self.dictionary)
        self.childrenMoveNames.append(legalMoves)
        self.childrenStateSeen.append(torch.zeros(len(legalMoves)))
        self.childrenStateWin.append(torch.zeros(len(legalMoves)))

        policy = ActionToArray.moveEvaluations(legalMoves, arrayBoard, prediction)
        self.childrenPolicyEval.append(policy)

        #value = ValueEvaluation.moveValueEvaluations(legalMoves, actualBoard, self.neuralNet)
        noValue = torch.zeros(len(legalMoves))
        self.childrenValueEval.append(noValue)
        end = time.time()
        print("ADD TIME:", end-start)

    def playout(self, round,
                explorationConstant=2**0.5,  # lower? will test more.
                notFromBeginning=False, arrayBoard=0, pythonBoard=0, plies=0, wCap=0, bCap=0,
                actuallyAPawn=0,
                noise=True,
                printPGN=True):  # Here is the information just for starting at a different position

        whiteParentStateDictionary = []
        whiteStateSeen = []
        whiteStateWin = []

        blackParentStateDictionary = []
        blackStateSeen = []
        blackStateWin = []

        tempBoard = ChessEnvironment()
        if notFromBeginning:
            tempBoard.arrayBoard = arrayBoard
            tempBoard.board = pythonBoard
            tempBoard.plies = plies
            tempBoard.whiteCaptivePieces = wCap
            tempBoard.blackCaptivePieces = bCap
            tempBoard.actuallyAPawn = actuallyAPawn
            tempBoard.updateNumpyBoards()

        depth = 0
        while tempBoard.result == 2 and depth < self.DEPTH_VALUE:
            depth += 1
            position = tempBoard.boardToString()
            if position not in self.dictionary:
                        # Create a new entry in the tree, if the state is not seen before.
                        state = torch.from_numpy(tempBoard.boardToState())
                        action = torch.zeros(1)

                        data = DoubleHeadDataset(state, action, action)
                        testLoader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=False)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        start = time.time()
                        for images, irrelevant1, irrelevant2 in testLoader:
                            images = images.to(device)
                            outputs = self.neuralNet(images)[0]
                        end = time.time()
                        print("BLAH:", end-start)
                        self.addPositionToMCTS(tempBoard.boardToString(),
                                               ActionToArray.legalMovesForState(tempBoard.arrayBoard,
                                                                                tempBoard.board),
                                               tempBoard.arrayBoard, outputs)
                        # find and make the preferred move
                        if noise:
                            noiseConstant = 0.15 / (1 * (1 + tempBoard.plies))  # should decrease this...
                        else:
                            noiseConstant = 0

                        if len(self.childrenStateWin) > 0:
                            _, index = (PUCT_Algorithm(self.childrenStateWin[len(self.childrenStateSeen) - 1],
                                                             self.childrenStateSeen[len(self.childrenStateSeen) - 1],
                                                             explorationConstant,
                                                             torch.sum(self.childrenStateSeen[
                                                                        len(self.childrenStateSeen) - 1]),
                                                             self.childrenValueEval[
                                                                            len(self.childrenStateSeen) - 1],
                                                             noiseEvals(self.childrenPolicyEval[len(self.childrenStateSeen) - 1],
                                                                        noiseConstant))).max(0)
                        else:
                            index = 0

                        move = self.childrenMoveNames[len(self.childrenStateSeen) - 1][index]

                        # print(move)
                        tempBoard.makeMove(move)

                        actionVector = torch.zeros(len(self.childrenMoveNames[len(self.childrenStateSeen) - 1]))
                        actionVector[index] = 1
            else:
                # find the directory of the move
                directory = self.dictionary[position]

                if noise:
                    noiseConstant = 0.6 / (2.5 * (1 + tempBoard.plies))
                else:
                    noiseConstant = 0

                _, index = (PUCT_Algorithm(self.childrenStateWin[directory],
                                                 self.childrenStateSeen[directory], explorationConstant,
                                                 torch.sum(self.childrenStateSeen[directory]),
                                                 self.childrenValueEval[directory],
                                                 noiseEvals(self.childrenPolicyEval[directory], noiseConstant)
                                                 )).max(0)
                move = self.childrenMoveNames[directory][index]

                # print(move)
                tempBoard.makeMove(move)

                # the move will have to be indexed correctly based on where the position is.
                actionVector = torch.zeros(len(self.childrenMoveNames[directory]))
                actionVector[index] = 1

            # add this into the actions chosen.
            if tempBoard.plies % 2 == 1:  # white has moved.
                whiteParentStateDictionary.append(position)
                whiteStateSeen.append(actionVector)
            else:  # black has moved
                blackParentStateDictionary.append(position)
                blackStateSeen.append(actionVector)

            # print(tempBoard.board)
            tempBoard.gameResult()

        if tempBoard.result == 1:  # white victory
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i])
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j] * 0)
        if tempBoard.result == -1:  # black victory
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i] * 0)
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j])
                # this is okay, because if the game is played til checkmate then
                # this ensures that the move count for both sides is equal.
        if tempBoard.result == 0:  # 'tis a tie
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i] * 0.5)
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j] * 0.5)

        if tempBoard.result == 2:  # game isn't played to very end
            winRate = ValueEvaluation.positionEval(tempBoard, self.neuralNet)
            # tempBoard.printBoard()
            # print(ActionToArray.legalMovesForState(tempBoard.arrayBoard, tempBoard.board))
            # if depth is not divisible by two then win rate is of opponent
            if depth % 2 == 0:
                if tempBoard.plies % 2 == 0:
                    # this means that we are evaluating white
                    for i in range(len(whiteStateSeen)):
                        whiteStateWin.append(whiteStateSeen[i] * winRate)
                    for j in range(len(blackStateSeen)):
                        blackStateWin.append(blackStateSeen[j] * (1-winRate))
                else:
                    # this means that we are evaluating black
                    for i in range(len(whiteStateSeen)):
                        whiteStateWin.append(whiteStateSeen[i] * (1-winRate))
                    for j in range(len(blackStateSeen)):
                        blackStateWin.append(blackStateSeen[j] * winRate)
            else:
                winRate = 1-winRate
                if tempBoard.plies % 2 == 1:
                    # this means that we are evaluating white
                    for i in range(len(whiteStateSeen)):
                        whiteStateWin.append(whiteStateSeen[i] * winRate)
                    for j in range(len(blackStateSeen)):
                        blackStateWin.append(blackStateSeen[j] * (1-winRate))
                else:
                    # this means that we are evaluating black
                    for i in range(len(whiteStateSeen)):
                        whiteStateWin.append(whiteStateSeen[i] * (1-winRate))
                    for j in range(len(blackStateSeen)):
                        blackStateWin.append(blackStateSeen[j] * winRate)

        # now, add the information into the MCTS database.
        for i in range(len(whiteStateSeen)):
            directory = self.dictionary[whiteParentStateDictionary[i]]
            self.childrenStateSeen[directory] = self.childrenStateSeen[directory] + whiteStateSeen[i]
            self.childrenStateWin[directory] = self.childrenStateWin[directory] + whiteStateWin[i]
        for i in range(len(blackStateSeen)):
            directory = self.dictionary[blackParentStateDictionary[i]]
            self.childrenStateSeen[directory] = self.childrenStateSeen[directory] + blackStateSeen[i]
            self.childrenStateWin[directory] = self.childrenStateWin[directory] + blackStateWin[i]

    def trainingPlayoutFromBeginning(self, runs, printPGN):
        for i in range(1, runs + 1):
            print("GAME", str(i))
            self.playout(str(i), printPGN=printPGN)

    def competitivePlayoutFromBeginning(self, runs, printPGN):
        for i in range(1, runs + 1):
            print("GAME", str(i))
            self.playout(str(i), noise=False, printPGN=printPGN)

    def trainingPlayoutsFromPosition(self, runs, sim):
        for i in range(runs):
            tempBoard = copy.deepcopy(sim)
            # playout from a certain position.
            self.playout(str(int(i + 1)), notFromBeginning=True, arrayBoard=tempBoard.arrayBoard,
                         pythonBoard=tempBoard.board,
                         plies=tempBoard.plies, wCap=tempBoard.whiteCaptivePieces, noise=True,
                         bCap=tempBoard.blackCaptivePieces, actuallyAPawn=tempBoard.actuallyAPawn,
                         explorationConstant=2**0.5, printPGN=False)

    def competitivePlayoutsFromPosition(self, runs, sim):
        for i in range(runs):
            print(i, "playouts finished.")
            tempBoard = copy.deepcopy(sim)
            # playout from a certain position.
            self.playout(str(int(i + 1)), notFromBeginning=True, arrayBoard=tempBoard.arrayBoard,
                         pythonBoard=tempBoard.board,
                         plies=tempBoard.plies, wCap=tempBoard.whiteCaptivePieces, explorationConstant=2**0.5,
                         bCap=tempBoard.blackCaptivePieces, noise=False, actuallyAPawn=tempBoard.actuallyAPawn,
                         printPGN=False)

            self.printSize()
            print(self.childrenMoveNames[self.dictionary[sim.boardToString()]])
            print(self.childrenStateWin[self.dictionary[sim.boardToString()]]/self.childrenStateSeen[self.dictionary[sim.boardToString()]])
            print(self.childrenStateSeen[self.dictionary[sim.boardToString()]])
            print(self.childrenPolicyEval[self.dictionary[sim.boardToString()]])
            print("Playouts:", torch.sum(self.childrenStateSeen[self.dictionary[sim.boardToString()]]))

testing = False
if testing:
    # TESTING
    legalMoves = ['a','b','c','d','e','f','g','h','i','j']
    nnPolicyPredictions = torch.rand(10)
    won = torch.rand(10)
    seen = torch.rand(10)
    seen[4]=0
    nnValuePredictions = torch.rand(10)
    final = PUCT_Algorithm(won,seen,2**0.5,torch.sum(seen), nnValuePredictions, noiseEvals(nnPolicyPredictions, 0.5))

    # THIS FINDS INDEX OF TOP VALUE
    _, indice = PUCT_Algorithm(won,seen,2**0.5,torch.sum(seen), nnValuePredictions,noiseEvals(nnPolicyPredictions, 0.5)).max(0)
    print(indice)
    print(legalMoves[indice])

MCTS = True
if MCTS:
    board = ChessEnvironment()
    search = MCTSTensor('/Users/gordon/Documents/CrazyhouseRL/New Networks/(MCTS)(8X256|8|8)(GPU)64fish.pt', 3)
    search.competitivePlayoutsFromPosition(20, board)

