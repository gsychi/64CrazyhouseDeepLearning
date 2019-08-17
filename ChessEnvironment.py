import time

import chess.variant
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessConvNet import ChessConvNet
import chess.pgn
import copy

class ChessEnvironment():

    def __init__(self):
        self.board = chess.variant.CrazyhouseBoard()  # this allows legal moves and all
        self.arrayBoard = [["r", "n", "b", "q", "k", "b", "n", "r"],
                           ["p", "p", "p", "p", "p", "p", "p", "p"],
                           [" ", " ", " ", " ", " ", " ", " ", " "],
                           [" ", " ", " ", " ", " ", " ", " ", " "],
                           [" ", " ", " ", " ", " ", " ", " ", " "],
                           [" ", " ", " ", " ", " ", " ", " ", " "],
                           ["P", "P", "P", "P", "P", "P", "P", "P"],
                           ["R", "N", "B", "Q", "K", "B", "N", "R"]]
        self.actuallyAPawn = np.zeros((8, 8))
        self.plies = 0
        # pawn, knight, bishop, rook, queen.
        self.whiteCaptivePieces = [0, 0, 0, 0, 0]
        self.blackCaptivePieces = [0, 0, 0, 0, 0]

        # This is required
        self.wPawnBoard = np.zeros((8, 8))
        self.bPawnBoard = np.zeros((8, 8))
        self.wKnightBoard = np.zeros((8, 8))
        self.bKnightBoard = np.zeros((8, 8))
        self.wBishopBoard = np.zeros((8, 8))
        self.bBishopBoard = np.zeros((8, 8))
        self.wRookBoard = np.zeros((8, 8))
        self.bRookBoard = np.zeros((8, 8))
        self.wQueenBoard = np.zeros((8, 8))
        self.bQueenBoard = np.zeros((8, 8))
        self.wKingBoard = np.zeros((8, 8))
        self.bKingBoard = np.zeros((8, 8))
        self.allBoards = np.zeros((1, 1))  # this will be refreshed anyway
        self.result = 2  # 2 denotes ongoing, 0 denotes draw, 1 denotes white win, -1 denotes black win
        self.stateFEN = chess.STARTING_FEN  # FEN of starting position
        self.gameStatus = "Game is in progress."

    def arrayBoardUpdate(self):
        # update boards
        for i in range(8):
            for j in range(8):
                if self.board.piece_at(8*i+j) != None:
                    self.arrayBoard[7-i][j] = self.board.piece_at(8*i+j).symbol()
                else:
                    self.arrayBoard[7-i][j] = ' '

        # update pockets
        self.whiteCaptivePieces = [0, 0, 0, 0, 0]
        self.blackCaptivePieces = [0, 0, 0, 0, 0]
        whitePocket = list(str(self.board.pockets[0]))
        blackPocket = list(str(self.board.pockets[1]))
        for i in range(len(whitePocket)):
            index = "pnbrq".find(whitePocket[i])
            self.whiteCaptivePieces[index] += 1
        for j in range(len(blackPocket)):
            index = "pnbrq".find(blackPocket[j])
            self.blackCaptivePieces[index] += 1


    def boardToFEN(self):
        self.stateFEN = self.board.fen()
        return self.stateFEN

    def boardToString(self):
        state = "0000000000000000000000000000000000000000000000000000000000000000"
        for i in range(8):
            for j in range(8):
                if self.arrayBoard[i][j] != " ":
                    direc = i * 8 + j
                    # change something
                    state = state[0:direc] + self.arrayBoard[i][j] + state[direc + 1:]
        captive = str(self.whiteCaptivePieces[0]) + str(self.whiteCaptivePieces[1]) \
                  + str(self.whiteCaptivePieces[2]) + str(self.whiteCaptivePieces[3]) \
                  + str(self.whiteCaptivePieces[4]) + str(self.blackCaptivePieces[0]) \
                  + str(self.blackCaptivePieces[1]) + str(self.blackCaptivePieces[2]) \
                  + str(self.blackCaptivePieces[3]) + str(self.blackCaptivePieces[4])

        turn = str(self.plies % 2)

        castling = '0000'
        if self.board.has_kingside_castling_rights(chess.WHITE):
            castling = '1000'
        if self.board.has_queenside_castling_rights(chess.WHITE):
            castling = castling[0] + '100'
        if self.board.has_kingside_castling_rights(chess.BLACK):
            castling = castling[0:2] + '10'
        if self.board.has_queenside_castling_rights(chess.BLACK):
            castling = castling[0:3] + '1'

        return state + captive + castling + turn + str(self.plies)

    def gameResult(self):
        if self.board.is_insufficient_material():
            self.result = 0
            self.gameStatus = "Draw."
        if self.board.is_stalemate():
            self.result = 0
            self.gameStatus = "Draw."
        if self.board.can_claim_draw():  # have to check for 3 fold soon
            self.result = 0
            self.gameStatus = "Draw."
        if self.board.is_checkmate():
            if self.plies % 2 == 0:
                # last move was black, therefore black won.
                self.result = -1
                self.gameStatus = "Black Victory"
            if self.plies % 2 == 1:
                self.result = 1
                self.gameStatus = "White Victory"

    def updateNumpyBoards(self):

        # Before updating states, one must refresh the boards.
        self.wPawnBoard = np.zeros((8, 8))
        self.bPawnBoard = np.zeros((8, 8))
        self.wKnightBoard = np.zeros((8, 8))
        self.bKnightBoard = np.zeros((8, 8))
        self.wBishopBoard = np.zeros((8, 8))
        self.bBishopBoard = np.zeros((8, 8))
        self.wRookBoard = np.zeros((8, 8))
        self.bRookBoard = np.zeros((8, 8))
        self.wQueenBoard = np.zeros((8, 8))
        self.bQueenBoard = np.zeros((8, 8))
        self.wKingBoard = np.zeros((8, 8))
        self.bKingBoard = np.zeros((8, 8))

        for i in range(8):
            for j in range(8):
                if self.arrayBoard[i][j] == "P":
                    self.wPawnBoard[i][j] = 1
                if self.arrayBoard[i][j] == "p":
                    self.bPawnBoard[i][j] = 1
                if self.arrayBoard[i][j] == "N":
                    self.wKnightBoard[i][j] = 1
                if self.arrayBoard[i][j] == "n":
                    self.bKnightBoard[i][j] = 1
                if self.arrayBoard[i][j] == "B":
                    self.wBishopBoard[i][j] = 1
                if self.arrayBoard[i][j] == "b":
                    self.bBishopBoard[i][j] = 1
                if self.arrayBoard[i][j] == "R":
                    self.wRookBoard[i][j] = 1
                if self.arrayBoard[i][j] == "r":
                    self.bRookBoard[i][j] = 1
                if self.arrayBoard[i][j] == "Q":
                    self.wQueenBoard[i][j] = 1
                if self.arrayBoard[i][j] == "q":
                    self.bQueenBoard[i][j] = 1
                if self.arrayBoard[i][j] == "K":
                    self.wKingBoard[i][j] = 1
                if self.arrayBoard[i][j] == "k":
                    self.bKingBoard[i][j] = 1
                # once all boards are done, concatenate them into the state
                self.allBoards = np.concatenate((self.wPawnBoard, self.wKnightBoard, self.wBishopBoard, self.wRookBoard,
                                                 self.wQueenBoard, self.wKingBoard,
                                                 self.bPawnBoard, self.bKnightBoard, self.bBishopBoard, self.bRookBoard,
                                                 self.bQueenBoard, self.bKingBoard, self.actuallyAPawn
                                                 ))

    def makeMove(self, move):
        if chess.Move.from_uci(move) in self.board.legal_moves:
            self.board.push(chess.Move.from_uci(move))

            # update boards
            for i in range(8):
                for j in range(8):
                    if self.board.piece_at(8*i+j) != None:
                        self.arrayBoard[7-i][j] = self.board.piece_at(8*i+j).symbol()
                    else:
                        self.arrayBoard[7-i][j] = ' '

                # update pockets
                self.whiteCaptivePieces = [0, 0, 0, 0, 0]
                self.blackCaptivePieces = [0, 0, 0, 0, 0]
                whitePocket = list(str(self.board.pockets[0]))
                blackPocket = list(str(self.board.pockets[1]))

                for k in range(len(whitePocket)):
                    index = "pnbrq".find(whitePocket[k])
                    self.whiteCaptivePieces[index] += 1
                for m in range(len(blackPocket)):
                    index = "pnbrq".find(blackPocket[m])
                    self.blackCaptivePieces[index] += 1

            self.updateNumpyBoards()
            self.plies += 1

    def printBoard(self):
        print(self.board)
        for i in range(8):
            print(self.arrayBoard[i])
        print(self.whiteCaptivePieces)
        print(self.blackCaptivePieces)

    def boardToState(self):

        captiveToBinary = np.zeros((16, 8))

        # Create copies of whiteCaptive and blackCaptive
        temp1 = np.copy(self.whiteCaptivePieces)
        temp2 = np.copy(self.blackCaptivePieces)

        # start off by updating pawns.
        for i in range(8):
            if temp1[0] > 0:
                captiveToBinary[0][i] = 1
                temp1[0] -= 1
            if temp2[0] > 0:
                captiveToBinary[4][i] = 1
                temp2[0] -= 1
        for i in range(8):
            if temp1[0] > 0:
                captiveToBinary[1][i] = 1
                temp1[0] -= 1
            if temp2[0] > 0:
                captiveToBinary[5][i] = 1
                temp2[0] -= 1

        # then, update knights, bishops, and rooks, and then queen.
        for i in range(4):
            if temp1[1] > 0:
                captiveToBinary[2][i] = 1
                temp1[1] -= 1
            if temp1[2] > 0:
                captiveToBinary[2][4 + i] = 1
                temp1[2] -= 1
            if temp1[3] > 0:
                captiveToBinary[3][i] = 1
                temp1[3] -= 1
            if temp1[4] > 0:
                captiveToBinary[3][4 + i] = 1
                temp1[4] -= 1
            if temp2[1] > 0:
                captiveToBinary[6][i] = 1
                temp2[1] -= 1
            if temp2[2] > 0:
                captiveToBinary[6][4 + i] = 1
                temp2[2] -= 1
            if temp2[3] > 0:
                captiveToBinary[7][i] = 1
                temp2[3] -= 1
            if temp2[4] > 0:
                captiveToBinary[7][4 + i] = 1
                temp2[4] -= 1

            # [7][6], [7][7] determine who is moving
            captiveToBinary[7][6], captiveToBinary[7][7] = (self.plies % 2), 1 - (self.plies % 2)

            if self.board.has_kingside_castling_rights(chess.WHITE):
                captiveToBinary[8][0] = 1
                captiveToBinary[8][1] = 1
            if self.board.has_queenside_castling_rights(chess.WHITE):
                captiveToBinary[8][2] = 1
                captiveToBinary[8][3] = 1
            if self.board.has_kingside_castling_rights(chess.BLACK):
                captiveToBinary[8][4] = 1
                captiveToBinary[8][5] = 1
            if self.board.has_kingside_castling_rights(chess.BLACK):
                captiveToBinary[8][6] = 1
                captiveToBinary[8][7] = 1

            if self.board.is_repetition(count=3):
                captiveToBinary[9][0] = 1
                #print("THREE TIME")
            elif self.board.is_repetition(count=2):
                captiveToBinary[9][1] = 1
                #print("TWO TIME")

        self.updateNumpyBoards()

        # perhaps work on adding 1s for spaces....
        return np.reshape(np.concatenate((self.allBoards, captiveToBinary)), (1, 15, 8, 8))


