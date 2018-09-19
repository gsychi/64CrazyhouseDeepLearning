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

def NetworkCompetitionWhite(network, playouts, round="1"):
    PGN = chess.pgn.Game()
    PGN.headers["Event"] = "Neural Network Comparison Test"
    PGN.headers["Site"] = "Cozy Computer Lounge"
    PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    PGN.headers["Round"] = round
    PGN.headers["Black"] = "Network: " + network.nameOfNetwork
    PGN.headers["White"] = "Human"
    PGN.headers["Variant"] = "Crazyhouse"

    sim = ChessEnvironment()
    while sim.result == 2:
        noiseVal = 0.6 / (6 * (sim.plies // 2 + 1))
        if sim.plies % 2 == 1:
            if playouts > 0:
                network.competitivePlayoutsFromPosition(playouts, sim)
            else:
                position = sim.boardToString()
                if position not in network.dictionary:
                    state = torch.from_numpy(sim.boardToState())
                    nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
                    testSet = MyDataset(state, nullAction)
                    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
                    with torch.no_grad():
                        for images, labels in generatePredic:
                            outputs = network.neuralNet(images)
                            network.addPositionToMCTS(sim.boardToString(),
                                                      ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                                       sim.board),
                                                      sim.arrayBoard, outputs)
            directory = network.dictionary[sim.boardToString()]
            index = np.argmax(
                MCTSCrazyhouse.PUCT_Algorithm(network.childrenStateWin[directory], network.childrenStateSeen[directory], 0.02,
                               np.sum(network.childrenStateSeen[directory]),
                               MCTSCrazyhouse.noiseEvals(network.childrenNNEvaluation[directory], noiseVal))
            )
            move = network.childrenMoveNames[directory][index]
            if chess.Move.from_uci(move) not in sim.board.legal_moves:
                move = ActionToArray.legalMovesForState(sim.arrayBoard, sim.board)[0]
            print(move)
            sim.makeMove(move)
            sim.gameResult()
        elif sim.plies % 2 == 0:
            # move is what human inputs
            move = input("Choose move: ")
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

def NetworkCompetitionBlack(network, playouts, round="1"):
    PGN = chess.pgn.Game()
    PGN.headers["Event"] = "Neural Network Comparison Test"
    PGN.headers["Site"] = "Cozy Computer Lounge"
    PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    PGN.headers["Round"] = round
    PGN.headers["Black"] = "Human"
    PGN.headers["White"] = "Network: " + network.nameOfNetwork
    PGN.headers["Variant"] = "Crazyhouse"


    sim = ChessEnvironment()
    while sim.result == 2:
        noiseVal = 0.6/(6*(sim.plies//2 + 1))
        if sim.plies % 2 == 0:
            if playouts > 0:
                network.competitivePlayoutsFromPosition(playouts, sim)
            else:
                position = sim.boardToString()
                if position not in network.dictionary:
                    state = torch.from_numpy(sim.boardToState())
                    nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
                    testSet = MyDataset(state, nullAction)
                    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
                    with torch.no_grad():
                        for images, labels in generatePredic:
                            outputs = network.neuralNet(images)
                            network.addPositionToMCTS(sim.boardToString(),
                                              ActionToArray.legalMovesForState(sim.arrayBoard,
                                                                               sim.board),
                                              sim.arrayBoard, outputs)
            directory = network.dictionary[sim.boardToString()]
            index = np.argmax(
                MCTSCrazyhouse.PUCT_Algorithm(network.childrenStateWin[directory], network.childrenStateSeen[directory], 0.02,
                               np.sum(network.childrenStateSeen[directory]),
                               MCTSCrazyhouse.noiseEvals(network.childrenNNEvaluation[directory], noiseVal))
            )
            move = network.childrenMoveNames[directory][index]
            if chess.Move.from_uci(move) not in sim.board.legal_moves:
                move = ActionToArray.legalMovesForState(sim.arrayBoard, sim.board)[0]
            print(move)
            sim.makeMove(move)
            sim.gameResult()
        elif sim.plies % 2 == 1:
            # move is what human inputs
            move = input("Choose move: ")
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


network = MCTS('sFULL1-5LAYER-24.pt')
NetworkCompetitionBlack(network, 0)
