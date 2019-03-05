# best net vs old net competition!
import datetime
import threading
import numpy as np
from ChessEnvironment import ChessEnvironment
import ActionToArray
import ChessConvNet
import _thread
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from MCTSCrazyhouse import MCTS
import copy
import chess.variant
import chess.pgn
import chess
import chess.svg
import MCTSCrazyhouse
import time
import ValueEvaluation

def NetworkCompetitionWhite(bestNet, playouts, round="1"):
    PGN = chess.pgn.Game()
    PGN.headers["Event"] = "Neural Network Comparison Test"
    PGN.headers["Site"] = "Cozy Computer Lounge"
    PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    PGN.headers["Round"] = round
    PGN.headers["White"] = "Network: " + bestNet.nameOfNetwork
    PGN.headers["Black"] = "You"
    PGN.headers["Variant"] = "crazyhouse"


    sim = ChessEnvironment()
    while sim.result == 2:
        noiseVal = 0.0 / (10 * (sim.plies // 2 + 1))
        if sim.plies % 2 == 0:
            if playouts > 0:
                start = time.time()
                bestNet.competitivePlayoutsFromPosition(playouts, sim)
                end = time.time()
                print(end-start)
            else:
                position = sim.boardToString()
                if position not in bestNet.dictionary:
                            image = torch.from_numpy(sim.boardToState())
                            outputs = bestNet.neuralNet(image)[0]
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


            # PRINT WIN PROBABILITY W/ MCTS?
            print("-----")
            print(move)
            print("Win Probability: {:.4f} %".format(100*ValueEvaluation.positionEval(sim, bestNet.neuralNet)))
            if playouts > 0 and bestNet.childrenStateSeen[directory][index] > 0:
                mctsWinRate = 100*bestNet.childrenStateWin[directory][index]/bestNet.childrenStateSeen[directory][index]
                print("MCTS Win Probability: {:.4f} %".format(mctsWinRate))
                totalWinRate = (100*ValueEvaluation.positionEval(sim, bestNet.neuralNet)+mctsWinRate)/2
                print("Total Win Probability: {:.4f} %".format(totalWinRate))
            print("-----")

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
        print("WHITE POCKET")
        print(sim.whiteCaptivePieces)
        print("BLACK POCKET")
        print(sim.blackCaptivePieces)
        

    if sim.result == 1:
        PGN.headers["Result"] = "1-0"
    if sim.result == 0:
        PGN.headers["Result"] = "1/2-1/2"
    if sim.result == -1:
        PGN.headers["Result"] = "0-1"

    print(PGN)

def NetworkCompetitionBlack(bestNet, playouts, round="1"):
    PGN = chess.pgn.Game()
    PGN.headers["Event"] = "Neural Network Comparison Test"
    PGN.headers["Site"] = "Cozy Computer Lounge"
    PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    PGN.headers["Round"] = round
    PGN.headers["White"] = "You"
    PGN.headers["Black"] = "Network: " + bestNet.nameOfNetwork
    PGN.headers["Variant"] = "crazyhouse"

    sim = ChessEnvironment()
    while sim.result == 2:
        noiseVal = 0.0 / (10*(sim.plies//2 + 1))
        if sim.plies % 2 == 1:
            if playouts > 0:
                start = time.time()
                bestNet.competitivePlayoutsFromPosition(playouts, sim)
                end = time.time()
                print(end-start)
            else:
                position = sim.boardToString()
                if position not in bestNet.dictionary:
                            image = torch.from_numpy(sim.boardToState())
                            outputs = bestNet.neuralNet(image)[0]
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


            # PRINT WIN PROBABILITY W/ MCTS?
            print("-----")
            print(move)
            print("Win Probability: {:.4f} %".format(100*ValueEvaluation.positionEval(sim, bestNet.neuralNet)))
            if playouts > 0 and bestNet.childrenStateSeen[directory][index]>0:
                mctsWinRate = 100*bestNet.childrenStateWin[directory][index]/bestNet.childrenStateSeen[directory][index]
                print("MCTS Win Probability: {:.4f} %".format(mctsWinRate))
                totalWinRate = (100*ValueEvaluation.positionEval(sim, bestNet.neuralNet)+mctsWinRate)/2
                print("Total Win Probability: {:.4f} %".format(totalWinRate))
            print("-----")

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
        print("WHITE POCKET")
        print(sim.whiteCaptivePieces)
        print("BLACK POCKET")
        print(sim.blackCaptivePieces)

    if sim.result == 1:
        PGN.headers["Result"] = "1-0"
    if sim.result == 0:
        PGN.headers["Result"] = "1/2-1/2"
    if sim.result == -1:
        PGN.headers["Result"] = "0-1"

    print(PGN)

# PLAY!
network = MCTS('Users/Gordon/Documents/CrazyhouseRL/New Networks/(MCTS)(8X256|8|8)(GPU)64fish.pt', 5)
NetworkCompetitionBlack(network, 100)
