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
ENGINE_DEPTH = 8
ENGINE_PLAYOUTS = 0
NOISE_INITIAL = 0.3
NOISE_DECAY = 1.4
ESTIMATED_NPS = 18

board = ChessEnvironment()
model = MCTS('/Users/gordon/Documents/CrazyhouseRL/New Networks/(MCTS)(8X256|8|8)(GPU)(V4)64fish.pt', ENGINE_DEPTH)

while True:
    command = input("")
    if command == "uci":
        print("id name 64\nid author Gordon Chi")
        print("option name UCI_Variant type combo default crazyhouse var crazyhouse")
        print("uciok")
    elif command.startswith("setoption"):
        settings = command[10:]
        if settings.__contains__("ENGINE_PLAYOUTS"):
            settings = int(settings[16:])
            ENGINE_PLAYOUTS = settings
        elif settings.__contains__("ENGINE_DEPTH"):
            settings = int(settings[13:])
            print(settings)
            model.ENGINE_DEPTH_VALUE = settings
        elif settings.__contains__("NETWORK"):
            settings = settings[8:]
            # switch to other test networks

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
        noiseVal = (NOISE_INITIAL*NOISE_DECAY) / (NOISE_DECAY * (board.plies // 2 + 1))

        # See if time management is needed.
        if not command.__contains__("infinite"):
            wtime = command.index("wtime")
            btime = command.index("btime")

            whiteTimeRemaining = int(int(command[wtime+6:btime-1])/1000)
            blackTimeRemaining = int(int(command[btime+6:])/1000)

            # TIME MANAGEMENT FOR WHITE:
            if board.plies % 2 == 0:
                # If it's the first move, spend roughly 10 seconds looking for the first move.
                # Only needed for lichess API.
                if board.plies == 0:
                    model.ENGINE_DEPTH_VALUE = 6
                    ENGINE_PLAYOUTS = 20
                else:
                    if whiteTimeRemaining < 10:
                        ENGINE_PLAYOUTS = 0
                    else:
                        # spend roughly 1/7 of available time
                        AVAILABLE_TIME = int(whiteTimeRemaining/7)
                        SEARCHABLE_NODES = ESTIMATED_NPS*AVAILABLE_TIME

                        # If over fifteen minutes on the clock...
                        if whiteTimeRemaining > 1500:
                            model.ENGINE_DEPTH_VALUE = 20
                        # If over five minutes on the clock...
                        if whiteTimeRemaining > 300:
                            model.ENGINE_DEPTH_VALUE = 15
                        # If over two minutes on the clock:
                        elif whiteTimeRemaining > 120:
                            model.ENGINE_DEPTH_VALUE = 10
                        # If over one minute on the clock
                        elif whiteTimeRemaining > 60:
                            model.ENGINE_DEPTH_VALUE = 8
                        else:
                            model.ENGINE_DEPTH_VALUE = 4

                        ENGINE_PLAYOUTS = int(SEARCHABLE_NODES/model.ENGINE_DEPTH_VALUE)

                        # Avoid engine from wasting time searching 1 or 2 playouts.
                        if ENGINE_PLAYOUTS < 3:
                            ENGINE_PLAYOUTS = 0

                        print("PLAYOUTS:", ENGINE_PLAYOUTS)
            # TIME MANAGEMENT FOR BLACK:
            else:
                # If it's the first move, spend roughly 10 seconds looking for the first move.
                if board.plies == 1:
                    model.ENGINE_DEPTH_VALUE = 6
                    ENGINE_PLAYOUTS = 20
                else:
                    if blackTimeRemaining < 10:
                        ENGINE_PLAYOUTS = 0
                    else:
                        # spend roughly 1/7 of available time
                        AVAILABLE_TIME = int(blackTimeRemaining/7)
                        SEARCHABLE_NODES = ESTIMATED_NPS*AVAILABLE_TIME

                        # If over fifteen minutes on the clock...
                        if blackTimeRemaining > 1500:
                            model.ENGINE_DEPTH_VALUE = 20
                        # If over five minutes on the clock...
                        if blackTimeRemaining > 300:
                            model.ENGINE_DEPTH_VALUE = 15
                        # If over two minutes on the clock:
                        elif blackTimeRemaining > 120:
                            model.ENGINE_DEPTH_VALUE = 10
                        # If over one minute on the clock
                        elif blackTimeRemaining > 60:
                            model.ENGINE_DEPTH_VALUE = 8
                        else:
                            model.ENGINE_DEPTH_VALUE = 4

                        ENGINE_PLAYOUTS = int(SEARCHABLE_NODES/model.ENGINE_DEPTH_VALUE)

                        # Avoid engine from wasting time searching 1 or 2 playouts.
                        if ENGINE_PLAYOUTS < 3:
                            ENGINE_PLAYOUTS = 0

                        print("PLAYOUTS:", ENGINE_PLAYOUTS)


        # START SEARCHING FOR MCTS TREE
        if ENGINE_PLAYOUTS > 0:
            start = time.time()
            model.competitivePlayoutsFromPosition(ENGINE_PLAYOUTS, board)
            end = time.time()
            TIME_SPENT = end-start
        else:
                position = board.boardToString()
                if position not in model.dictionary:
                            state = torch.from_numpy(board.boardToState())
                            model.neuralNet.eval()
                            outputs = model.neuralNet(state)[0]
                            model.neuralNet.eval()
                            if ENGINE_PLAYOUTS > 0:
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

        # MAKE A MOVE
        if ENGINE_PLAYOUTS > 0:
            index = np.argmax(model.childrenStateSeen[directory])
            """
                                index = np.argmax(
                                    MCTSCrazyhouse.PUCT_Algorithm(model.childrenStateWin[directory], model.childrenStateSeen[directory], 1,
                                                   np.sum(model.childrenStateSeen[directory]),
                                                                  model.childrenValueEval[directory],
                                                   MCTSCrazyhouse.noiseEvals(model.childrenPolicyEval[directory], noiseVal))
                                )
            """
        else:
            index = np.argmax(MCTSCrazyhouse.noiseEvals(model.childrenPolicyEval[directory], noiseVal))
        move = model.childrenMoveNames[directory][index]
        if chess.Move.from_uci(move) not in board.board.legal_moves:
            move = ActionToArray.legalMovesForState(board.arrayBoard, board.board)[0]
        print("bestmove " + move)

        # PRINT LOG
        if ENGINE_PLAYOUTS > 0:
            NODES_PER_SECOND = int(round((model.ENGINE_DEPTH_VALUE*ENGINE_PLAYOUTS)/TIME_SPENT, 0))
            MCTS_WIN_RATE = model.childrenStateWin[directory][index]/model.childrenStateSeen[directory][index]
            WINRATE_LOG = str(int(round(ValueEvaluation.objectivePositionEvalMCTS(board, model.neuralNet, MCTS_WIN_RATE), 4)))
            print("Win Probability (0-1):", MCTS_WIN_RATE)
        else:
            NODES_PER_SECOND = 0
            WINRATE_LOG = str(int(round(ValueEvaluation.objectivePositionEval(board, model.neuralNet), 4)))
        if ENGINE_PLAYOUTS > 0:
            print("info depth", model.ENGINE_DEPTH_VALUE, "score cp", WINRATE_LOG, "time", int(TIME_SPENT*1000),
                  "nodes", ENGINE_PLAYOUTS*model.ENGINE_DEPTH_VALUE, "nps", NODES_PER_SECOND, "pv", move)
        else:
            print("info depth 0 score cp", WINRATE_LOG, "time 1 nodes 1 nps", NODES_PER_SECOND, "pv", move)


        board.makeMove(move)
        #print(board.board)
        board.gameResult()






