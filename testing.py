import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessConvNet import ChessConvNet
from PolicyDataset import PolicyDataset
import ChessResNet
import h5py
from ChessEnvironment import ChessEnvironment


board = ChessEnvironment()
board.makeMove("e2e4")
board.makeMove("d7d5")
board.makeMove("e4d5")
board.makeMove("d8d5")
board.makeMove("b1c3")
board.makeMove("g8f6")
board.makeMove("c3d5")
board.makeMove("f6d5")
a=(board.boardToState().flatten())

print(board.boardToFEN())
car = ChessEnvironment()
car.board.set_fen(board.boardToFEN())
car.arrayBoardUpdate()
car.updateNumpyBoards()
b=(board.boardToState().flatten())

for i in range(len(b)):
    if a[i] != b[i]:
        print("not identical match")
