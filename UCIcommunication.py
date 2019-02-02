#!/usr/local/bin/python3.6
import numpy as np
from ChessEnvironment import ChessEnvironment
from DoubleHeadDataset import DoubleHeadDataset
import ActionToArray
import ChessConvNet
import torch
import _thread
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

# PARAMETERS
depth = 16
playouts = 0

board = ChessEnvironment()
model = MCTS('/Volumes/back up/CrazyhouseRL/New Networks/(MCTS)(6X128|4|8)(V1)64fish.pt', depth)

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
        elif settings.__contains__("network"):
            settings = settings[8:]
            # switch to other test networks
            """
            if settings.__contains__("stockfish"):
                model = MCTS('/Users/gordon/Documents/CrazyhouseRL/New Networks/stock-8X256-PV.pt', depth)
            elif settings.__contains__("test"):
                model = MCTS('/Users/gordon/Documents/CrazyhouseRL/New Networks/NEW-8X256-PV.pt', depth)
            """
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
            moveWordIndex = command.find("moves")
            notFromStarting = False
            if moveWordIndex == -1:
                command = command[4:]
            else:
                notFromStarting = True
                moves = command[moveWordIndex+6:].split(" ")
                command = command[4:moveWordIndex]
            print(command)
            if notFromStarting:
                print(moves)
            board = ChessEnvironment()
            board.board.set_fen(command)
            if notFromStarting:
                for i in range(len(moves)):
                        board.makeMove(moves[i])
            board.arrayBoardUpdate()
            board.updateNumpyBoards()

    # make a move
    elif command.startswith("go"):
        noiseVal = 4 / (8 * (board.plies // 2 + 1))
        if playouts > 0:
            model.competitivePlayoutsFromPosition(playouts, board)
        else:
                position = board.boardToString()
                if position not in model.dictionary:
                            state = torch.from_numpy(board.boardToState())
                            model.neuralNet.eval()
                            outputs = model.neuralNet(state)[0]
                            model.neuralNet.eval()
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
        print("Win Rate:",100*round(ValueEvaluation.positionEval(board, model.neuralNet), 4),"%")
        print("info depth 1 score cp", str(int(1000*round(ValueEvaluation.objectivePositionEval(board, model.neuralNet), 3))), "time 1 nodes 1 nps 1 pv", move)

        board.makeMove(move)
        print(board.board)
        board.gameResult()






