import chess.variant
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessEnvironment import ChessEnvironment
from ChessConvNet import ChessConvNet
from MyDataset import MyDataset
import ActionToArray
import numpy_indexed as npi
import pathlib

pgnGames = list(pathlib.Path('lichessdatabase').glob('*.pgn'))
listOfMoves = []
for i in range(len(pgnGames)):
    pgn = open(pgnGames[i])
    for k in range(40):  # 190,000 assures all games are looked at.
        try:
            game = chess.pgn.read_game(pgn)
            whiteElo = int(game.headers["WhiteElo"])
            blackElo = int(game.headers["BlackElo"])
            benchmark = 2300
            if whiteElo >= benchmark and blackElo >= benchmark:
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

f = open("2018games2000.txt", "w+")
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
        if board.board.legal_moves.count() != len(ActionToArray.legalMovesForState(board.arrayBoard, board.board)):
            print("ERROR!")

        board.makeMove(listOfMoves[j][i])
        # add it to database
        inList.append(state)
        outList.append(action)


    print(board.board)
    board.gameResult()
    print(board.gameStatus)
    print(len(inList))
    print(len(outList))
    print(str(int(j + 1)), "out of ", len(listOfMoves), "parsed.")

# all games are parsed, now convert list into array
inputs = np.zeros((len(inList), 1, 32, 28))

outputs = np.zeros((len(outList), 4504))

i = 0
while len(inList) > 0:
    inputs[i] = inList[len(inList)-1][0]
    outputs[i] = outList[len(inList)-1][0]
    inList.pop()
    outList.pop()
    i += 1

print(inputs.shape)
print(outputs.shape)


"""
Below we convert our weighted probabilities into the logit function 
so that it works under the sigmoid function later on. 
The NN gets confused by this so we will comment it out for now.
"""
# outputs = (outputs/1.0002)+0.0001
# outputs = np.log((outputs/(1-outputs)))


np.save("Training Data/SMALLmasterInputs.npy", inputs)
np.save("Training Data/SMALLmasterOutputs.npy", outputs)