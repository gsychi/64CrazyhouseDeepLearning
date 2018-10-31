# best net vs old net competition!
import datetime

import numpy as np
from ChessEnvironment import ChessEnvironment
from MyDataset import MyDataset
import ActionToArray
import ChessConvNet
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from MCTSCrazyhouse import MCTS
#from EnsembleMCTSCrazyhouse import EnsembleMCTS
import copy
import chess.variant
import chess.pgn
import chess
import MCTSCrazyhouse
import time

def NetworkCompetitionWhite(bestNet, testingNet, playouts, round="1"):
    score = 0
    PGN = chess.pgn.Game()
    PGN.headers["Event"] = "Neural Network Comparison Test"
    PGN.headers["Site"] = "Cozy Computer Lounge"
    PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    PGN.headers["Round"] = round
    PGN.headers["White"] = "Network: " + bestNet.nameOfNetwork
    PGN.headers["Black"] = "Network: " + testingNet.nameOfNetwork
    PGN.headers["Variant"] = "Crazyhouse"


    sim = ChessEnvironment()
    while sim.result == 2:
        noiseVal = 3.0 / (10 * (sim.plies // 2 + 1))
        if sim.plies % 2 == 0:
            if playouts > 0:
                bestNet.competitivePlayoutsFromPosition(playouts, sim)
            else:
                position = sim.boardToString()
                if position not in bestNet.dictionary:
                    state = torch.from_numpy(sim.boardToState())
                    nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
                    testSet = MyDataset(state, nullAction)
                    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
                    with torch.no_grad():
                        for images, labels in generatePredic:
                            bestNet.neuralNet.eval()
                            outputs = bestNet.neuralNet(images)
                            bestNet.addPositionToMCTS(sim.boardToString(),
                                                      ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                                       sim.board),
                                                      sim.arrayBoard, outputs)
            directory = bestNet.dictionary[sim.boardToString()]
            index = np.argmax(
                MCTSCrazyhouse.PUCT_Algorithm(bestNet.childrenStateWin[directory], bestNet.childrenStateSeen[directory], 0.02,
                               np.sum(bestNet.childrenStateSeen[directory]),
                               MCTSCrazyhouse.noiseEvals(bestNet.childrenNNEvaluation[directory], noiseVal))
            )
            move = bestNet.childrenMoveNames[directory][index]
            if chess.Move.from_uci(move) not in sim.board.legal_moves:
                move = ActionToArray.legalMovesForState(sim.arrayBoard, sim.board)[0]
            print(move)
            sim.makeMove(move)
            sim.gameResult()
        elif sim.plies % 2 == 1:
            if playouts > 0:
                testingNet.competitivePlayoutsFromPosition(playouts, sim)
            else:
                position = sim.boardToString()
                if position not in testingNet.dictionary:
                    state = torch.from_numpy(sim.boardToState())
                    nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
                    testSet = MyDataset(state, nullAction)
                    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
                    with torch.no_grad():
                        for images, labels in generatePredic:
                            testingNet.neuralNet.eval()
                            outputs = testingNet.neuralNet(images)
                            testingNet.addPositionToMCTS(sim.boardToString(),
                                                         ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                                          sim.board),
                                                         sim.arrayBoard, outputs)
            directory = testingNet.dictionary[sim.boardToString()]
            index = np.argmax(
                MCTSCrazyhouse.PUCT_Algorithm(testingNet.childrenStateWin[directory], testingNet.childrenStateSeen[directory], 0.02,
                                              np.sum(testingNet.childrenStateSeen[directory]),
                                              MCTSCrazyhouse.noiseEvals(testingNet.childrenNNEvaluation[directory],
                                                                        noiseVal))
            )
            move = testingNet.childrenMoveNames[directory][index]
            if chess.Move.from_uci(move) not in sim.board.legal_moves:
                move = ActionToArray.legalMovesForState(sim.arrayBoard, sim.board)[0]
            print(move)
            sim.makeMove(move)
            sim.gameResult()

        if sim.plies == 1:
            node = PGN.add_variation(chess.Move.from_uci(move))
        else:
            node = node.add_variation(chess.Move.from_uci(move))

        print(sim.board)

    if sim.result == 1:
        PGN.headers["Result"] = "1-0"
    if sim.result == 0:
        PGN.headers["Result"] = "1/2-1/2"
        score = 0.5
    if sim.result == -1:
        PGN.headers["Result"] = "0-1"
        score = 1

    print(PGN)
    return score

def NetworkCompetitionBlack(bestNet, testingNet, playouts, round="1"):
    score = 0
    PGN = chess.pgn.Game()
    PGN.headers["Event"] = "Neural Network Comparison Test"
    PGN.headers["Site"] = "Cozy Computer Lounge"
    PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    PGN.headers["Round"] = round
    PGN.headers["White"] = "Network: " + testingNet.nameOfNetwork
    PGN.headers["Black"] = "Network: " + bestNet.nameOfNetwork
    PGN.headers["Variant"] = "Crazyhouse"


    sim = ChessEnvironment()
    while sim.result == 2:
        noiseVal = 3.0/(10*(sim.plies//2 + 1))
        if sim.plies % 2 == 1:
            if playouts > 0:
                bestNet.competitivePlayoutsFromPosition(playouts, sim)
            else:
                position = sim.boardToString()
                if position not in bestNet.dictionary:
                    state = torch.from_numpy(sim.boardToState())
                    nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
                    testSet = MyDataset(state, nullAction)
                    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
                    with torch.no_grad():
                        for images, labels in generatePredic:
                            bestNet.neuralNet.eval()
                            outputs = bestNet.neuralNet(images)
                            bestNet.addPositionToMCTS(sim.boardToString(),
                                              ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board),
                                              sim.arrayBoard, outputs)
            directory = bestNet.dictionary[sim.boardToString()]
            index = np.argmax(
                MCTSCrazyhouse.PUCT_Algorithm(bestNet.childrenStateWin[directory], bestNet.childrenStateSeen[directory], 0.02,
                               np.sum(bestNet.childrenStateSeen[directory]),
                               MCTSCrazyhouse.noiseEvals(bestNet.childrenNNEvaluation[directory], noiseVal))
            )
            move = bestNet.childrenMoveNames[directory][index]
            if chess.Move.from_uci(move) not in sim.board.legal_moves:
                move = ActionToArray.legalMovesForState(sim.arrayBoard, sim.board)[0]
            print(move)
            sim.makeMove(move)
            sim.gameResult()
        elif sim.plies % 2 == 0:
            if playouts > 0:
                testingNet.competitivePlayoutsFromPosition(playouts, sim)
            else:
                position = sim.boardToString()
                if position not in testingNet.dictionary:
                    state = torch.from_numpy(sim.boardToState())
                    nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
                    testSet = MyDataset(state, nullAction)
                    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
                    with torch.no_grad():
                        for images, labels in generatePredic:
                            testingNet.neuralNet.eval()
                            outputs = testingNet.neuralNet(images)
                            testingNet.addPositionToMCTS(sim.boardToString(),
                                              ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board),
                                              sim.arrayBoard, outputs)
            directory = testingNet.dictionary[sim.boardToString()]
            index = np.argmax(
                MCTSCrazyhouse.PUCT_Algorithm(testingNet.childrenStateWin[directory], testingNet.childrenStateSeen[directory], 0.02,
                                              np.sum(testingNet.childrenStateSeen[directory]),
                                              MCTSCrazyhouse.noiseEvals(testingNet.childrenNNEvaluation[directory],
                                                                        noiseVal))
            )
            move = testingNet.childrenMoveNames[directory][index]
            if chess.Move.from_uci(move) not in sim.board.legal_moves:
                move = ActionToArray.legalMovesForState(sim.arrayBoard, sim.board)[0]
            print(move)
            sim.makeMove(move)
            sim.gameResult()

        if sim.plies == 1:
            node = PGN.add_variation(chess.Move.from_uci(move))
        else:
            node = node.add_variation(chess.Move.from_uci(move))

        print(sim.board)

    if sim.result == 1:
        PGN.headers["Result"] = "1-0"
        score = 1
    if sim.result == 0:
        PGN.headers["Result"] = "1/2-1/2"
        score = 0.5
    if sim.result == -1:
        PGN.headers["Result"] = "0-1"

    print(PGN)
    return score

# run 400 games.
def bestNetworkTest(bestNet, testingNet, games, playouts, clearAfterEachRound=False):
    score = 0
    for i in range(games):
        round = str(int(i+1))
        if i%2==0:
            score += NetworkCompetitionWhite(bestNet, testingNet, playouts, round=round)
            print("Win Rate:", str(100*(score/(i+1))),"%")
        elif i%2==1:
            score += NetworkCompetitionBlack(bestNet, testingNet, playouts, round=round)
            print("Win Rate:", str(100*(score/(i+1))),"%")

        # Make sure that the playouts are updated after
        if clearAfterEachRound:
            bestNet.clearInformation()
            testingNet.clearInformation()

    print("Score: ", str(score), ":", str((games - score)))
    if score > (games/2):
        print("BEST: ", bestNet.nameOfNetwork)
        print("TEST: ", testingNet.nameOfNetwork)
        print("new network is better than old network")
        return True
    else:
        print("BEST: ", bestNet.nameOfNetwork)
        print("TEST: ", testingNet.nameOfNetwork)
        return False


testing = True
if testing:
    best = MCTS("7 Layer k=32 Models/v5-1706to1809.pt")
    newNet = MCTS('1705to1810-b50000.pt')
    #newNet = MCTS('experimentalPoisson.pt')
    print(bestNetworkTest(best, newNet, 200, 0))
