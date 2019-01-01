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
import copy
import chess.variant
import chess.pgn
import chess
import MCTSCrazyhouse
import time
import ValueEvaluation

def NetworkCompetitionWhite(bestNet, playouts, round="1"):
    score = 0
    PGN = chess.pgn.Game()
    PGN.headers["Event"] = "Neural Network Comparison Test"
    PGN.headers["Site"] = "Cozy Computer Lounge"
    PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    PGN.headers["Round"] = round
    PGN.headers["White"] = "Network: " + bestNet.nameOfNetwork
    PGN.headers["Black"] = "You"
    PGN.headers["Variant"] = "Crazyhouse"


    sim = ChessEnvironment()
    while sim.result == 2:
        print("Win Probability:", ValueEvaluation.positionEval(sim, bestNet.valueNet))
        noiseVal = 0.0 / (10 * (sim.plies // 2 + 1))
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
            legal = False
            while not legal:
                move = input("Enter move: ")
                if len(move) == 4 or len(move) == 5:
                    if chess.Move.from_uci(move) in sim.board.legal_moves:
                        legal = True
                    else:
                        print("Illegal move! Try again:")
                else:
                    print("Illegal move! Try again:")
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
    if sim.result == -1:
        PGN.headers["Result"] = "0-1"

    print(PGN)

def NetworkCompetitionBlack(bestNet, playouts, round="1"):
    score = 0
    PGN = chess.pgn.Game()
    PGN.headers["Event"] = "Neural Network Comparison Test"
    PGN.headers["Site"] = "Cozy Computer Lounge"
    PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    PGN.headers["Round"] = round
    PGN.headers["White"] = "You"
    PGN.headers["Black"] = "Network: " + bestNet.nameOfNetwork
    PGN.headers["Variant"] = "Crazyhouse"

    sim = ChessEnvironment()
    while sim.result == 2:
        print("Win Probability:", ValueEvaluation.positionEval(sim, bestNet.valueNet))
        noiseVal = 0.0 / (10*(sim.plies//2 + 1))
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
            legal = False
            while not legal:
                move = input("Enter move: ")
                if len(move) == 4 or len(move) == 5:
                    if chess.Move.from_uci(move) in sim.board.legal_moves:
                        legal = True
                    else:
                        print("Illegal move! Try again:")
                else:
                    print("Illegal move! Try again:")
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
    if sim.result == -1:
        PGN.headers["Result"] = "0-1"

    print(PGN)

# PLAY!
network = MCTS('New Networks/18011810-ckpt8-POLICY.pt', 'New Networks/18011810-VALUE.pt', 3)
NetworkCompetitionWhite(network, 0)
