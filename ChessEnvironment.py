import chess.variant
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessConvNet import ChessConvNet
import ActionToArray
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
        return state + captive + castling + turn

    def gameResult(self):
        if self.board.is_insufficient_material():
            self.result = 0
            self.gameStatus = "Draw."
        if self.board.is_stalemate():
            self.result = 0
            self.gameStatus = "Draw."
        if self.board.can_claim_draw():
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
        if chess.Move.from_uci(move) not in self.board.legal_moves:
            print(move)
            print("Illegal Move!")
            print(self.board)
            print(self.arrayBoard)

            # make some random move
            legalMoves = ActionToArray.legalMovesForState(self.arrayBoard, self.board)
            illegalMove = True
            while illegalMove:
                for i in range(len(legalMoves)):
                    if chess.Move.from_uci(legalMoves[i]) in self.board.legal_moves:
                        move = legalMoves[i]
                        illegalMove = False
        if chess.Move.from_uci(move) in self.board.legal_moves:
            self.board.push(chess.Move.from_uci(move))
            # update numpy board too - split the move and find coordinates! see old chess java work.
            rowNames = "abcdefgh"
            if move[1] != "@":
                initialRow = 8 - int(move[1])  # for e2d4, move[1] returns 2
            else:
                initialRow = 0
            initialCol = int(rowNames.find(move[0]))  # for e2d4, move[1] returns e
            finalRow = 8 - int(move[3])  # for e2d4, move[3] returns 4
            finalCol = int(rowNames.find(move[2]))  # for e2d4, move[2] returns d
            # SPECIAL MOVE 1: CASTLING. MAKE SURE THAT THE PIECE IN QUESTION IS A KING!!!
            if move == "e1g1" and self.arrayBoard[7][4] == "K" and self.arrayBoard[7][7] == "R":
                self.arrayBoard[7][4] = " "
                self.arrayBoard[7][7] = " "
                self.arrayBoard[7][5] = "R"
                self.arrayBoard[7][6] = "K"
            elif move == "e8g8" and self.arrayBoard[0][4] == "k" and self.arrayBoard[0][7] == "r":
                self.arrayBoard[0][4] = " "
                self.arrayBoard[0][7] = " "
                self.arrayBoard[0][5] = "R"
                self.arrayBoard[0][6] = "K"
            elif move == "e8c8" and self.arrayBoard[0][4] == "k" and self.arrayBoard[0][0] == "r":
                self.arrayBoard[0][0] = " "
                self.arrayBoard[0][1] = " "
                self.arrayBoard[0][4] = " "
                self.arrayBoard[0][2] = "K"
                self.arrayBoard[0][3] = "R"
            elif move == "e1c1" and self.arrayBoard[7][4] == "K" and self.arrayBoard[7][0] == "R":
                self.arrayBoard[7][0] = " "
                self.arrayBoard[7][1] = " "
                self.arrayBoard[7][4] = " "
                self.arrayBoard[7][2] = "K"
                self.arrayBoard[7][3] = "R"
            # SPECIAL MOVE 2: EN PASSANT
            # check if the capture square is empty and there is a pawn on the same row but different column
            # white en passant
            elif self.arrayBoard[initialRow][initialCol] == "P" and self.arrayBoard[initialRow][finalCol] == "p" and \
                    self.arrayBoard[finalRow][finalCol] == " ":
                # print("WHITE EN PASSANT")
                self.arrayBoard[initialRow][initialCol] = " "
                self.arrayBoard[finalRow][finalCol] = "P"
                self.arrayBoard[initialRow][finalCol] = " "
                self.whiteCaptivePieces[0] += 1
            # black en passant
            elif self.arrayBoard[initialRow][initialCol] == "p" and self.arrayBoard[initialRow][finalCol] == "P" and \
                    self.arrayBoard[finalRow][finalCol] == " ":
                # print("BLACK EN PASSANT")
                self.arrayBoard[initialRow][initialCol] = " "
                self.arrayBoard[finalRow][finalCol] = "p"
                self.arrayBoard[initialRow][finalCol] = " "
                self.blackCaptivePieces[0] += 1
            elif "PRNBQ".find(move[0]) == -1:
                # update the board
                temp = self.arrayBoard[finalRow][finalCol]
                self.arrayBoard[finalRow][finalCol] = self.arrayBoard[initialRow][initialCol]
                self.arrayBoard[initialRow][initialCol] = " "

                # move around the actuallyAPawn stuff too.
                wasAPawn = self.actuallyAPawn[finalRow][finalCol]
                self.actuallyAPawn[finalRow][finalCol] = self.actuallyAPawn[initialRow][initialCol]
                self.actuallyAPawn[initialRow][initialCol] = 0

                # this is for promotion
                if len(move) == 5:
                    if self.plies % 2 == 0:
                        self.arrayBoard[finalRow][finalCol] = move[4].upper()
                    if self.plies % 2 == 1:
                        self.arrayBoard[finalRow][finalCol] = move[4].lower()
                    self.actuallyAPawn[finalRow][finalCol] = 1

                # add pieces to captured area
                if wasAPawn == 0:  # 0 means it is normal.
                    whiteCaptured = "pnbrq".find(temp)
                    blackCaptured = "PNBRQ".find(temp)
                    if whiteCaptured > -1:
                        self.whiteCaptivePieces[whiteCaptured] += 1
                    if blackCaptured > -1:
                        self.blackCaptivePieces[blackCaptured] += 1
                if wasAPawn == 1:  # 1 means that the piece in question was once a pawn.
                    if self.plies % 2 == 0:
                        self.whiteCaptivePieces[0] += 1
                    if self.plies % 2 == 1:
                        self.blackCaptivePieces[0] += 1

            else:
                # this is when a captured piece is put back on the board

                # update the captive pieces
                placed = "PNBRQ".find(move[0])
                if self.plies % 2 == 0:
                    self.whiteCaptivePieces[placed] -= 1
                if self.plies % 2 == 1:
                    self.blackCaptivePieces[placed] -= 1

                # update the board.
                rowNames = "abcdefgh"
                placedRow = 8 - int(move[3])
                placedCol = int(rowNames.find(move[2]))

                if self.plies % 2 == 0:
                    self.arrayBoard[placedRow][placedCol] = move[0]
                if self.plies % 2 == 1:
                    self.arrayBoard[placedRow][placedCol] = move[0].lower()

            # once everything is done, update move count
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

        self.updateNumpyBoards()

        # perhaps work on adding 1s for spaces....
        return np.reshape(np.concatenate((self.allBoards, captiveToBinary)), (1, 15, 8, 8))


