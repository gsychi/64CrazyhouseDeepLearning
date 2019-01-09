#!/usr/local/bin/python3.6
import numpy as np
from ChessEnvironment import ChessEnvironment
from DoubleHeadDataset import DoubleHeadDataset
from MyDataset import MyDataset
import ActionToArray
import ChessConvNet
import torch
import torch.nn as nn
import sys
import torch.utils.data as data_utils
from MCTSCrazyhouse import MCTS
import MCTSCrazyhouse
import copy
import chess.variant
import chess.pgn
import chess
import time
import ValueEvaluation
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

board = ChessEnvironment()
model = MCTS('/Users/gordon/Documents/CrazyhouseRL/New Networks/smallnet.pt', 3)
playouts = 0

while True:
    command = input("")
    if command == "uci":
        print("id name 64\nid author Gordon Chi")
        print("option name UCI_Variant type combo default crazyhouse var crazyhouse")
        print("uciok")
    elif command.startswith("setoption"):
        settings = command[10:]
        if settings.__contains__("playouts"):
            settings = int(settings[9:])
            playouts = settings
        elif settings.__contains__("depth"):
            settings = int(settings[6:])
            model.DEPTH_VALUE = settings

    elif command == "isready":
        print("readyok")
    elif command == "print":
        print(board.printBoard())
    elif command == "quit":
        sys.exit(0)
    elif command.startswith("position"):
        command = command[9:]
        if command.__contains__("startpos"):
            command = command[9:]
            board = ChessEnvironment()
            if command.__contains__("moves"):
                # these are all the moves
                moves = command[6:].split(" ")
                for i in range(len(moves)):
                    board.makeMove(moves[i])
                #print(board.board)

        if command.__contains__("fen"):
            command = command[4:]
            board = ChessEnvironment()
            board.board.set_fen(command)
            board.arrayBoardUpdate()
            board.updateNumpyBoards()

    elif command.startswith("position"):
        print("YAY")
        if command.__contains__("startpos"):
            print("ASDHJKASHDJKH")
            command = command[9:]
            print("SDFJ")
            print(command)
        if command.__contains__("moves"):
            command = command[6:]
            while len(command)>0:
                # check where next ' ' is.
                indice = command.index(' ')
                move = command[0:indice]
                board.makeMove(move)
                command = command[indice+1:]

    # make a move
    elif command.startswith("go"):
        noiseVal = 3.0 / (10 * (board.plies // 2 + 1))
        if playouts > 0:
                model.competitivePlayoutsFromPosition(playouts, board)
        else:
                position = board.boardToString()
                if position not in model.dictionary:
                    state = torch.from_numpy(board.boardToState())
                    nullAction = torch.from_numpy(np.zeros(1))  # this will not be used, is only a filler
                    testSet = DoubleHeadDataset(state, nullAction, nullAction)
                    generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
                    with torch.no_grad():
                        for images, labels1, labels2 in generatePredic:
                            model.neuralNet.eval()
                            outputs = model.neuralNet(images)[0]
                            if playouts > 0:
                                model.addPositionToMCTS(board.boardToString(),
                                              ActionToArray.legalMovesForState(board.arrayBoard,
                                                                               board.board),
                                              board.arrayBoard, outputs, board)
                            else:
                                model.dictionary[board.boardToString()] = len(model.dictionary)
                                policy = ActionToArray.moveEvaluations(ActionToArray.legalMovesForState(board.arrayBoard,
                                                                               board.board),
                                                                       board.arrayBoard, outputs)
                                model.childrenPolicyEval.append(policy)
                                model.childrenMoveNames.append(ActionToArray.legalMovesForState(board.arrayBoard,
                                                                               board.board))
        directory = model.dictionary[board.boardToString()]
        if playouts > 0:
                                index = np.argmax(
                                    MCTSCrazyhouse.PUCT_Algorithm(model.childrenStateWin[directory], model.childrenStateSeen[directory], 1,
                                                   np.sum(model.childrenStateSeen[directory]),
                                                                  model.childrenValueEval[directory],
                                                   MCTSCrazyhouse.noiseEvals(model.childrenPolicyEval[directory], noiseVal))
                                )
        else:
            index = np.argmax(MCTSCrazyhouse.noiseEvals(model.childrenPolicyEval[directory], noiseVal))
        move = model.childrenMoveNames[directory][index]
        if chess.Move.from_uci(move) not in board.board.legal_moves:
            move = ActionToArray.legalMovesForState(board.arrayBoard, board.board)[0]
        print("bestmove " + move)
        print("info depth 1 score cp", str(int(1000*round(ValueEvaluation.objectivePositionEval(board, model.neuralNet), 3))), "time 1 nodes 1 nps 1 pv", move)

        board.makeMove(move)
        print(board.board)
        board.gameResult()






