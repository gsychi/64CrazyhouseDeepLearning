"""
Creates full database of human data with three arrays:

Inputs: 15x8x8 images of the chessboard
Value Output: -1 to 1 evaluation of board (-1 is a win for black, 1 is a win for white)
Policy Output: Probabilities from 0 to 1 of making a move.
"""

import chess.variant
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessEnvironment import ChessEnvironment
from ChessConvNet import ChessConvNet
from MyDataset import MyDataset
import ActionToArray
import pathlib
import h5py

pgnGames = list(pathlib.Path('lichessdatabase').glob('*.pgn'))
listOfMoves = []
listOfResults = []
for i in range(2, 3): #len(pgnGames)):
    pgn = open(pgnGames[i])
    for k in range(190000):  # 190,000 assures all games are looked at.
        try:
            game = chess.pgn.read_game(pgn)
            whiteElo = int(game.headers["WhiteElo"])
            blackElo = int(game.headers["BlackElo"])
            result = str(game.headers["Result"])
            if result == "1-0":
                result = 1
            elif result == "0-1":
                result = -1
            else:
                result = 0

            benchmark = 2000
            if whiteElo >= benchmark and blackElo >= benchmark:
                print("Index: ", k)
                print(whiteElo)
                print(blackElo)
                board = game.board()
                singleGame = []
                for move in game.main_line():
                    board.push(move)
                    singleGame.append(move.uci())
                listOfMoves.append(singleGame)
                listOfResults.append(result)
                print(pgnGames[i])
                # print(listOfResults)
        except:
            print("", end="")

f = open("Training Data/201805games2000.txt", "w+")
for i in range(len(listOfMoves)):
    print(listOfMoves[i], ",")
    f.write(str(listOfMoves[i])+",\n")
f.close()


inList = []
outList = []
actionList = []

for j in range(len(listOfMoves)):
    board = ChessEnvironment()
    for i in range(len(listOfMoves[j])):
        state = board.boardToState()
        value = listOfResults[j]
        action = ActionToArray.moveArray(listOfMoves[j][i], board.arrayBoard)
        for k in range(320, 384):
            action[0][k] = 0
        if board.board.legal_moves.count() != len(ActionToArray.legalMovesForState(board.arrayBoard, board.board)):
            print("ERROR!")

        # make move
        board.makeMove(listOfMoves[j][i])

        # add to database
        inList.append(state)
        outList.append(value)
        actionList.append(np.argmax(action))

    print(board.board)
    board.gameResult()
    print(board.gameStatus)
    print(len(inList))
    print(len(outList))
    print(len(actionList))
    print(str(int(j + 1)), "out of ", len(listOfMoves), "parsed.")

# all games are parsed, now convert list into array
inputs = np.zeros((len(inList), 15, 8, 8))
valueOutputs = np.zeros(len(outList))
policyOutputs = np.zeros(len(actionList))

i = 0
while len(inList) > 0:
    inputs[i] = inList[len(inList)-1][0]
    valueOutputs[i] = outList[len(outList)-1]
    policyOutputs[i] = actionList[len(actionList)-1]
    inList.pop()
    outList.pop()
    actionList.pop()
    i += 1

print(valueOutputs)
print(policyOutputs)

print(inputs.shape)
print(valueOutputs.shape)
print(policyOutputs.shape)


with h5py.File('Training Data/18-02Inputs.h5', 'w') as hf:
    hf.create_dataset("Inputs", data=inputs, compression='gzip', compression_opts=9)
with h5py.File('Training Data/18-02PolicyOutputs.h5', 'w') as hf:
    hf.create_dataset("Outputs", data=policyOutputs, compression='gzip', compression_opts=9)
with h5py.File('Training Data/18-02ValueOutputs.h5', 'w') as hf:
    hf.create_dataset("Outputs", data=valueOutputs, compression='gzip', compression_opts=9)


