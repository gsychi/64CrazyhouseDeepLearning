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

def isThereMate(board, moves, engine):
    info = engine.analyse(board.board, chess.engine.Limit(time=0.005))
    if str(info["score"])[0] == "#":
        #print("mate found")
        move = str(info["pv"][0])
        # find where this move is in the possible actions
        for i in range(len(moves)):
            if moves[i] == move:
                #print("found index")
                return i
    return None


dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# PARAMETERS
ENGINE_DEPTH = 8
ENGINE_PLAYOUTS = 0
NOISE_INITIAL = 0.5
NOISE_DECAY = 1.22
ESTIMATED_NPS = 6

board = ChessEnvironment()
model = MCTS('/Users/gordon/Documents/CrazyhouseRL/New Networks/(MCTS)(12X256|16|8)(GPU)64fish.pt', ENGINE_DEPTH)

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
            model.DEPTH_VALUE = settings
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
            command = command.split(" ")

            if len(command) > 4:  # this means that there is time for both users!
                whiteTimeRemaining = int(int(command[2])/1000)
                blackTimeRemaining = int(int(command[4])/1000)
            else:
                whiteTimeRemaining = int(int(command[2])/1000)
                blackTimeRemaining = int(int(command[2])/1000)

            # TIME MANAGEMENT FOR WHITE:
            if board.plies % 2 == 0:
                # If it's the first move, spend roughly 10 seconds looking for the first move.
                # Only needed for lichess API.
                if board.plies == 0:
                    model.DEPTH_VALUE = 6
                    ENGINE_PLAYOUTS = 10
                else:
                    if whiteTimeRemaining < 20:
                        ENGINE_PLAYOUTS = 0
                    else:
                        # spend roughly 1/7 of available time
                        if board.plies > 20:
                            AVAILABLE_TIME = int(whiteTimeRemaining/7.5)
                        elif board.plies > 14:
                            AVAILABLE_TIME = int(whiteTimeRemaining/10)
                        elif board.plies > 6:
                            AVAILABLE_TIME = int(whiteTimeRemaining/15)
                        else:
                            AVAILABLE_TIME = 2

                        SEARCHABLE_NODES = ESTIMATED_NPS*AVAILABLE_TIME

                        # If over fifteen minutes on the clock
                        if whiteTimeRemaining > 750:
                            model.DEPTH_VALUE = 20
                        # If over five minutes on the clock
                        elif whiteTimeRemaining > 300:
                            model.DEPTH_VALUE = 15
                        # If over two minutes on the clock:
                        elif whiteTimeRemaining > 120:
                            model.DEPTH_VALUE = 12
                        # If over one minute on the clock:
                        elif whiteTimeRemaining > 60:
                            model.DEPTH_VALUE = 10
                        else:
                            model.DEPTH_VALUE = 4

                        ENGINE_PLAYOUTS = int(SEARCHABLE_NODES/model.DEPTH_VALUE)

                        # Avoid engine from wasting time searching 1 or 2 playouts.
                        if ENGINE_PLAYOUTS < 3:
                            ENGINE_PLAYOUTS = 0

                        print("PLAYOUTS:", ENGINE_PLAYOUTS)
            # TIME MANAGEMENT FOR BLACK:
            else:
                # If it's the first move, spend roughly 10 seconds looking for the first move.
                if board.plies == 1:
                    model.DEPTH_VALUE = 6
                    ENGINE_PLAYOUTS = 10
                else:
                    if blackTimeRemaining < 20:
                        ENGINE_PLAYOUTS = 0
                    else:
                        # spend roughly 1/7 of available time
                        if board.plies > 21:
                            AVAILABLE_TIME = int(blackTimeRemaining/7.5)
                        elif board.plies > 15:
                            AVAILABLE_TIME = int(blackTimeRemaining/10)
                        elif board.plies > 7:
                            AVAILABLE_TIME = int(blackTimeRemaining/15)
                        else:
                            AVAILABLE_TIME = 2
                        
                        SEARCHABLE_NODES = ESTIMATED_NPS*AVAILABLE_TIME

                        # If over fifteen minutes on the clock
                        if blackTimeRemaining > 750:
                            model.DEPTH_VALUE = 20
                        # If over five minutes on the clock
                        elif blackTimeRemaining > 300:
                            model.DEPTH_VALUE = 15
                        # If over two minutes on the clock
                        elif blackTimeRemaining > 120:
                            model.DEPTH_VALUE = 12
                        # If over one minute on the clock
                        elif blackTimeRemaining > 60:
                            model.DEPTH_VALUE = 10
                        else:
                            model.DEPTH_VALUE = 4

                        ENGINE_PLAYOUTS = int(SEARCHABLE_NODES/model.DEPTH_VALUE)

                        # Avoid engine from wasting time searching 1 or 2 playouts.
                        if ENGINE_PLAYOUTS < 3:
                            ENGINE_PLAYOUTS = 0

                        print("PLAYOUTS:", ENGINE_PLAYOUTS)


        # START SEARCHING FOR MCTS TREE
        if ENGINE_PLAYOUTS > 0:
            start = time.time()
            print("CHOSEN DEPTH:",model.DEPTH_VALUE)
            model.competitivePlayoutsFromPosition(ENGINE_PLAYOUTS, board)
            end = time.time()
            TIME_SPENT = end-start
            directory = model.dictionary[board.boardToString()]
            if board.plies > 10 or board.plies < 2:
                index = np.argmax(model.childrenStateSeen[directory])
            else:
                index = np.argmax(MCTSCrazyhouse.noiseEvals(model.childrenPolicyEval[directory], noiseVal))

            move = model.childrenMoveNames[directory][index]
        else:
            state = torch.from_numpy(board.boardToState())

            # moves in a position
            moveNames = ActionToArray.legalMovesForState(board.arrayBoard, board.board)

            mate = isThereMate(board, moveNames, model.matefinder)
            if mate != None:
                index = mate
                print("I see mate!")
            else:
                model.neuralNet.eval()
                outputs = model.neuralNet(state)[0]

                policyScore = ActionToArray.moveEvaluations(moveNames, board.arrayBoard, outputs)
                noise = (np.random.rand(len(policyScore)) * 2 * noiseVal) - (noiseVal)
                index = np.argmax(policyScore+noise)

            move = moveNames[index]

        if chess.Move.from_uci(move) not in board.board.legal_moves:
            move = ActionToArray.legalMovesForState(board.arrayBoard, board.board)[0]
        print("bestmove " + move)

        # PRINT LOG
        if ENGINE_PLAYOUTS > 0:
            NODES_PER_SECOND = int(round((model.DEPTH_VALUE*ENGINE_PLAYOUTS)/TIME_SPENT, 0))
            MCTS_WIN_RATE = model.childrenStateWin[directory][index]/model.childrenStateSeen[directory][index]
            try:
                WINRATE_LOG = str(int(round(ValueEvaluation.objectivePositionEvalMCTS(board, model.neuralNet, MCTS_WIN_RATE), 4)))
            except:
                 WINRATE_LOG = 0.5
            print("Win Probability (0-1):", MCTS_WIN_RATE)
        else:
            NODES_PER_SECOND = 0
            WINRATE_LOG = str(int(round(ValueEvaluation.objectivePositionEval(board, model.neuralNet), 4)))
        if ENGINE_PLAYOUTS > 0:
            print("info depth", model.DEPTH_VALUE, "score cp", WINRATE_LOG, "time", int(TIME_SPENT*1000),
                  "nodes", ENGINE_PLAYOUTS*model.DEPTH_VALUE, "nps", NODES_PER_SECOND, "pv", move)
        else:
            print("info depth 0 score cp", WINRATE_LOG, "time 1 nodes 1 nps", NODES_PER_SECOND, "pv", move)


        board.makeMove(move)
        #print(board.board)
        board.gameResult()






