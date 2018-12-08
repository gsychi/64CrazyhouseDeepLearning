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
for i in range(0, 1): #len(pgnGames)):
    pgn = open(pgnGames[i])
    for k in range(190000):  # 190,000 assures all games are looked at.
        try:
            game = chess.pgn.read_game(pgn)
            whiteElo = int(game.headers["WhiteElo"])
            blackElo = int(game.headers["BlackElo"])
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
                print(pgnGames[i])
        except:
            print("", end="")

f = open("Training Data/201805games2000.txt", "w+")
for i in range(len(listOfMoves)):
    print(listOfMoves[i], ",")
    f.write(str(listOfMoves[i])+",\n")
f.close()

inList = []
outList = []

for j in range(len(listOfMoves)):
    board = ChessEnvironment()
    for i in range(len(listOfMoves[j])):
        state = board.boardToState()
        action = ActionToArray.moveArray(listOfMoves[j][i], board.arrayBoard)
        for k in range(320, 384):
            action[0][k] = 0
        if board.board.legal_moves.count() != len(ActionToArray.legalMovesForState(board.arrayBoard, board.board)):
            print("ERROR!")

        board.makeMove(listOfMoves[j][i])
        # add it to database
        inList.append(state)
        outList.append(np.argmax(action))


    print(board.board)
    board.gameResult()
    print(board.gameStatus)
    print(len(inList))
    print(len(outList))
    print(str(int(j + 1)), "out of ", len(listOfMoves), "parsed.")

# all games are parsed, now convert list into array
inputs = np.zeros((len(inList), 15, 8, 8))
outputs = np.zeros(len(outList))

i = 0
while len(inList) > 0:
    inputs[i] = inList[len(inList)-1][0]
    outputs[i] = outList[len(inList)-1]
    inList.pop()
    outList.pop()
    i += 1

print(outputs)

print(inputs.shape)
print(outputs.shape)


"""
Below we convert our weighted probabilities into the logit function 
so that it works under the sigmoid function later on. 
The NN gets confused by this so we will comment it out for now.
"""
# outputs = (outputs/1.0002)+0.0001
# outputs = np.log((outputs/(1-outputs)))

with h5py.File('Training Data/18-01Inputs.h5', 'w') as hf:
    hf.create_dataset("Inputs", data=inputs, compression='gzip', compression_opts=9)
with h5py.File('Training Data/18-01Outputs.h5', 'w') as hf:
    hf.create_dataset("Outputs", data=outputs, compression='gzip', compression_opts=9)


