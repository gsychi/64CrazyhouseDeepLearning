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
import ValueEvaluation

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
        print("Win Probability:", ValueEvaluation.positionEval(sim, bestNet.valueNet))
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
                            bestNet.policyNet.eval()
                            outputs = bestNet.policyNet(images)
                            if playouts > 0:
                                bestNet.addPositionToMCTS(sim.boardToString(),
                                                      ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                                       sim.board),
                                                      sim.arrayBoard, outputs, sim)
                            else:
                                bestNet.dictionary[sim.boardToString()] = len(bestNet.dictionary)
                                policy = ActionToArray.moveEvaluations(ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board),
                                                                       sim.arrayBoard, outputs)
                                bestNet.childrenMoveNames.append(ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board))
                                bestNet.childrenPolicyEval.append(policy)

            directory = bestNet.dictionary[sim.boardToString()]
            if playouts > 0:
                index = np.argmax(
                    MCTSCrazyhouse.PUCT_Algorithm(bestNet.childrenStateWin[directory], bestNet.childrenStateSeen[directory], 1,
                                   np.sum(bestNet.childrenStateSeen[directory]),
                                   bestNet.childrenValueEval[directory],
                                   MCTSCrazyhouse.noiseEvals(bestNet.childrenPolicyEval[directory], noiseVal))
                )
            else:
                index = np.argmax(MCTSCrazyhouse.noiseEvals(bestNet.childrenPolicyEval[directory], noiseVal))
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
                            testingNet.policyNet.eval()
                            outputs = testingNet.policyNet(images)
                            if playouts > 0:
                                testingNet.addPositionToMCTS(sim.boardToString(),
                                                         ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                                          sim.board),
                                                         sim.arrayBoard, outputs, sim)
                            else:
                                testingNet.dictionary[sim.boardToString()] = len(testingNet.dictionary)
                                policy = ActionToArray.moveEvaluations(ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board),
                                                                       sim.arrayBoard, outputs)
                                testingNet.childrenMoveNames.append(ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board))
                                testingNet.childrenPolicyEval.append(policy)

            directory = testingNet.dictionary[sim.boardToString()]
            if playouts > 0:
                index = np.argmax(
                    MCTSCrazyhouse.PUCT_Algorithm(testingNet.childrenStateWin[directory], testingNet.childrenStateSeen[directory], 1,
                                                  np.sum(testingNet.childrenStateSeen[directory]),
                                                  testingNet.childrenValueEval[directory],
                                                  MCTSCrazyhouse.noiseEvals(testingNet.childrenPolicyEval[directory],
                                                                            noiseVal))
                )
            else:
                index = np.argmax(MCTSCrazyhouse.noiseEvals(testingNet.childrenPolicyEval[directory], noiseVal))
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
        print("Win Probability:", ValueEvaluation.positionEval(sim, bestNet.valueNet))
        noiseVal = 3.0 /(10*(sim.plies//2 + 1))
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
                            bestNet.policyNet.eval()
                            outputs = bestNet.policyNet(images)
                            if playouts > 0:
                                bestNet.addPositionToMCTS(sim.boardToString(),
                                              ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board),
                                              sim.arrayBoard, outputs, sim)
                            else:
                                bestNet.dictionary[sim.boardToString()] = len(bestNet.dictionary)
                                policy = ActionToArray.moveEvaluations(ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board),
                                                                       sim.arrayBoard, outputs)
                                bestNet.childrenPolicyEval.append(policy)
                                bestNet.childrenMoveNames.append(ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board))

            directory = bestNet.dictionary[sim.boardToString()]
            if playouts > 0:
                index = np.argmax(
                    MCTSCrazyhouse.PUCT_Algorithm(bestNet.childrenStateWin[directory], bestNet.childrenStateSeen[directory], 1,
                                   np.sum(bestNet.childrenStateSeen[directory]),
                                                  bestNet.childrenValueEval[directory],
                                   MCTSCrazyhouse.noiseEvals(bestNet.childrenPolicyEval[directory], noiseVal))
                )
            else:
                index = np.argmax(MCTSCrazyhouse.noiseEvals(bestNet.childrenPolicyEval[directory], noiseVal))
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
                            testingNet.policyNet.eval()
                            outputs = testingNet.policyNet(images)
                            if playouts > 0:
                                testingNet.addPositionToMCTS(sim.boardToString(),
                                              ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board),
                                              sim.arrayBoard, outputs, sim)
                            else:
                                testingNet.dictionary[sim.boardToString()] = len(testingNet.dictionary)
                                policy = ActionToArray.moveEvaluations(ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board),
                                                                       sim.arrayBoard, outputs)
                                testingNet.childrenPolicyEval.append(policy)
                                testingNet.childrenMoveNames.append(ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board))

            directory = testingNet.dictionary[sim.boardToString()]
            if playouts > 0:
                index = np.argmax(
                    MCTSCrazyhouse.PUCT_Algorithm(testingNet.childrenStateWin[directory], testingNet.childrenStateSeen[directory], 1,
                                                  np.sum(testingNet.childrenStateSeen[directory]),
                                                  testingNet.childrenValueEval[directory],
                                                  MCTSCrazyhouse.noiseEvals(testingNet.childrenPolicyEval[directory],
                                                                            noiseVal))
                )
            else:
                index = np.argmax(MCTSCrazyhouse.noiseEvals(testingNet.childrenPolicyEval[directory], noiseVal))
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
    best = MCTS("New Networks/18011810-ckpt2-POLICY.pt", "Old Networks/18011805-VALUE.pt", 8)
    newNet = MCTS("New Networks/18011810-ARCH10X128-POLICY.pt", "Old Networks/18011805-VALUE.pt", 8)
    print(bestNetworkTest(best, newNet, 1000, 0))
