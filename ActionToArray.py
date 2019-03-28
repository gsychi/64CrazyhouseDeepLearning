"""
This is quite an important function, as it maps each action that the computer chooses
"""
import numpy as np
import chess.variant
import copy
import time
from scipy.interpolate import interp1d
from ChessEnvironment import ChessEnvironment

def legalMovesFromSquare(directory2, board, pythonChessBoard):

    rowNames = "abcdefgh"
    initialRow = rowNames[directory2 % 8]
    initialCol = str(8 - directory2 // 8)

    # find out what piece is being moved.
    pieceBeingMoved = board[directory2 // 8][directory2 % 8]
    # print("Piece moving:")
    # print(pieceBeingMoved)

    # create plane for all moves.
    # for that specific move, and see if that move is legal.

    possibleMovePlane = np.zeros((8, 8))

    # stuff for king
    if pieceBeingMoved.upper() == "K":
        for k in range(-1, 2):
            for l in range(-1, 2):
                i, j = (directory2 // 8) + k, (directory2 % 8) + l
                if i > -1 and j > -1 and i < 8 and j < 8:
                    possibleMovePlane[i][j] = 1

        # castling
        i, j = (directory2 // 8), (directory2 % 8)
        if j == 4 and (i == 0 or i == 7):
            possibleMovePlane[i][j-2] = 1
            possibleMovePlane[i][j+2] = 1

    # stuff for white pawn
    if pieceBeingMoved == "P":
        i, j = (directory2 // 8), (directory2 % 8)
        if i == 6:  # first move, move two steps forward
            possibleMovePlane[i - 2][j] = 1
        for k in range(-1, 2):
            if -1 < (j + k) < 8:
                possibleMovePlane[i - 1][j + k] = 1

    # stuff for white pawn
    if pieceBeingMoved == "p":
        i, j = (directory2 // 8), (directory2 % 8)
        if i == 1:  # first move, move two steps forward
            possibleMovePlane[i + 2][j] = 1
        for k in range(-1, 2):
            if -1 < (j + k) < 8:
                possibleMovePlane[i + 1][j + k] = 1

    # stuff for knight
    if pieceBeingMoved.upper() == "N":
        i, j = directory2 // 8, directory2 % 8

        # how do for loops work in python for +=2...
        try:
            possibleMovePlane[i + 2][j + 1] = 1
        except:
            print("", end='')
        try:
            if j - 1 > -1:
                possibleMovePlane[i + 2][j - 1] = 1
        except:
            print("", end='')
        try:
            possibleMovePlane[i + 1][j + 2] = 1
        except:
            print("", end='')
        try:
            if j - 2 > -1:
                possibleMovePlane[i + 1][j - 2] = 1
        except:
            print("", end='')
        try:
            if i - 2 > -1:
                possibleMovePlane[i - 2][j + 1] = 1
        except:
            print("", end='')
        try:
            if i - 2 > -1 and j - 1 > -1:
                possibleMovePlane[i - 2][j - 1] = 1
        except:
            print("", end='')
        try:
            if i - 1 > -1:
                possibleMovePlane[i - 1][j + 2] = 1
        except:
            print("", end='')
        try:
            if i - 1 > -1 and j - 2 > -1:
                possibleMovePlane[i - 1][j - 2] = 1
        except:
            print("", end='')

    # straight movement [for rook and queen]
    if pieceBeingMoved.upper() == "R" or pieceBeingMoved.upper() == "Q":
        for i in range(8):
            possibleMovePlane[i][directory2 % 8] = 1
            possibleMovePlane[directory2 // 8][i] = 1

    # diagonal movement...again would benefit from understanding for loops [for bishop and queen]
    if pieceBeingMoved.upper() == "B" or pieceBeingMoved.upper() == "Q":
        for k in range(2):
            for l in range(2):
                incI = (2 * k) - 1
                incJ = (2 * l) - 1
                i, j = directory2 // 8, directory2 % 8
                while -1 < i < 8 and -1 < j < 8:
                    possibleMovePlane[i][j] = 1
                    i += incI
                    j += incJ

    #print(possibleMovePlane)

    # now that the possible move plane is created, create a list of moves that are legal for that piece.
    # use the python-board to see if the move is indeed legal. If so, add it to the list.
    finalLegalMoves = []
    for i in range(8):
        for j in range(8):
            # check each square
            finalRow = rowNames[j]
            finalCol = str(8 - i)
            if possibleMovePlane[i][j] == 1:
                promotionPieces = ["n", "b", "r", "q"]
                # for white pawn
                if pieceBeingMoved == "P" and directory2 < 16:  # can be promoted
                    for k in range(len(promotionPieces)):
                        possibleMoveOption = initialRow + initialCol + finalRow + finalCol + promotionPieces[k]
                        if chess.Move.from_uci(possibleMoveOption) in pythonChessBoard.legal_moves:
                            finalLegalMoves.append(possibleMoveOption)
                # for black pawn
                if pieceBeingMoved == "p" and directory2 > 47:  # can be promoted
                    for k in range(len(promotionPieces)):
                        possibleMoveOption = initialRow + initialCol + finalRow + finalCol + promotionPieces[k]
                        if chess.Move.from_uci(possibleMoveOption) in pythonChessBoard.legal_moves:
                            finalLegalMoves.append(possibleMoveOption)
                # for not pawns
                else:
                    possibleMoveOption = initialRow + initialCol + finalRow + finalCol
                    if chess.Move.from_uci(possibleMoveOption) in pythonChessBoard.legal_moves:
                        finalLegalMoves.append(possibleMoveOption)

    # all legal moves for that piece is found! what shall it be...
    #print(finalLegalMoves)
    return finalLegalMoves

# this already accounts for the side that is moving
def moveablePieces(board, pythonChessBoard):
    moveablePieces = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            direc = i*8 + j
            if len(legalMovesFromSquare(direc, board, pythonChessBoard)) > 0:
                moveablePieces[i][j] = 1
    return moveablePieces

def legalMovesForState(board, pythonChessBoard):
    legalMoves = []
    for i in range(8):
        for j in range(8):
            direc = i*8 + j
            legalMoves = legalMoves + legalMovesFromSquare(direc, board, pythonChessBoard)

            # need to account for dropped pieces as well
            pieces = ["P", "N", "B", "R", "Q"]
            for k in range(5):
                rowNames = "abcdefgh"
                finalRow = rowNames[j]
                finalCol = str(8 - i)

                if board[i][j] == " ":
                    move = pieces[k] + "@" +finalRow + finalCol
                    if chess.Move.from_uci(move) in pythonChessBoard.legal_moves:
                        legalMoves.append(move)

    return legalMoves

def moveablePiecesPenalty(board, pythonChessBoard):
    moveablePieces = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            direc = i*8 + j
            if len(legalMovesFromSquare(direc, board, pythonChessBoard)) == 0:
                moveablePieces[i][j] = -100000
    return moveablePieces

# move is a string, turns into array.
def moveArray(move, board):
    #start = time.time()
    placedPlane = np.zeros((5, 8, 8))  # pawn, knight, bishop, rook, queen.
    movePlane = np.zeros((8, 7, 8, 8))  # direction (N, NE, E, SE, S, SW, W, NW), squares.
    knightMovePlane = np.zeros((8, 8, 8))  # direction ([1, 2],[2, 1],[2, -1],[1, -2],[-1, -2],[-2, -1],[-2, 1],[-1, 2])
    underPromotion = np.zeros((9, 8))  # this can be a 8x8 plane, but for now we will not. Knight, Bishop, Rook

    if "PRNBQ".find(move[0]) == -1:
        #print("MOVE TO ARRAY: NORMAL MOVE")
        rowNames = "abcdefgh"
        # identify how far the piece has moved, and how far it will be moving.
        if "PBRQK".find(board[8 - int(move[1])][int(rowNames.find(move[0]))].upper()) != -1:
            # print("its a", board[8 - int(move[1])][int(rowNames.find(move[0]))])
            if len(move) == 5:
                directory = "nbr".find(move[4].lower())  # .lower() just in case
                if directory != -1:
                    if move[0] == move[2]:
                        underPromotion[directory][int(rowNames.find(move[2]))] = 1
                    elif move[0] > move[2]:
                        underPromotion[3+directory][int(rowNames.find(move[2]))] = 1
                    elif move[2] > move[0]:
                        underPromotion[6+directory][int(rowNames.find(move[2]))] = 1
                else:
                    if int(move[3]) == 8:  # white queen promotion
                        columnMovement = int(rowNames.find(move[2])) - int(
                            rowNames.find(move[0]))  # positive = east, negative = west [EAST = +1, WEST = -1]
                        if columnMovement == 1:
                            movePlane[1][0][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
                        if columnMovement == 0:
                            movePlane[0][0][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
                        if columnMovement == -1:
                            movePlane[-1][0][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
                    if int(move[3]) == 1: # black queen promotion
                        columnMovement = int(rowNames.find(move[2])) - int(
                            rowNames.find(move[0]))  # positive = east, negative = west [EAST = +1, WEST = -1]
                        if columnMovement == 1:
                            movePlane[3][0][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
                        if columnMovement == 0:
                            movePlane[4][0][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
                        if columnMovement == -1:
                            movePlane[5][0][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
                    # find direction of promotion
                    # movement is one.
                    # change the plane

            else:
                rowMovement = int(move[3]) - int(move[1])  # positive = north, negative = south [NORTH = 0, SOUTH = 4]
                columnMovement = int(rowNames.find(move[2])) - int(
                    rowNames.find(move[0]))  # positive = east, negative = west [EAST = +1, WEST = -1]

                directory = 999
                if rowMovement > 0:  # NORTH
                    directory = 0
                    if columnMovement > 0:  # NORTH-EAST
                        directory = 1
                    if columnMovement < 0:  # NORTH-WEST
                        directory = 7
                elif rowMovement < 0:
                    directory = 4
                    if columnMovement > 0:  # SOUTH-EAST
                        directory = 3
                    if columnMovement < 0:  # SOUTH-WEST
                        directory = 5
                elif rowMovement == 0:
                    if columnMovement > 0:  # EAST:
                        directory = 2
                    if columnMovement < 0:  # WEST
                        directory = 6

                magnitude = max(abs(rowMovement), abs(columnMovement)) - 1
                movePlane[directory][magnitude][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
        elif board[8 - int(move[1])][int(rowNames.find(move[0]))] == " ":
            print("not legal move")  # don't do anything
        else:
            # print("its a knight")
            columnMovement = int(move[3]) - int(move[1])  # positive = north, negative = south [NORTH = 0, SOUTH = 4]
            rowMovement = int(rowNames.find(move[2])) - int(
                rowNames.find(move[0]))  # positive = east, negative = west [EAST = +1, WEST = -1]
            directory = 999
            if rowMovement == 1:
                if columnMovement == 2:
                    directory = 0
                elif columnMovement == -2:
                    directory = 3
            elif rowMovement == 2:
                if columnMovement == 1:
                    directory = 1
                elif columnMovement == -1:
                    directory = 2
            elif rowMovement == -2:
                if columnMovement == 1:
                    directory = 6
                elif columnMovement == -1:
                    directory = 5
            elif rowMovement == -1:
                if columnMovement == 2:
                    directory = 7
                elif columnMovement == -2:
                    directory = 4
            knightMovePlane[directory][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
    else:
        # print("PLACED PIECE")
        rowNames = "abcdefgh"
        placedPlane["PNBRQ".find(move[0])][8 - int(move[3])][int(rowNames.find(move[2]))] = 1

    """
    REMOVE REDUNDANT SQUARES...
    STILL IN EDITING.
    """
    # return the result.
    placedPlane = placedPlane.flatten()
    placedPlane = np.concatenate((placedPlane[8:56], placedPlane[64:320]), axis=None)
    movePlane = movePlane.flatten()
    # keep NW corner
    movePlane = movePlane[0:(64*7*7)+64*6+1]
    movePlane = np.concatenate((movePlane[:((64*7*7)+64*5)], movePlane[(64*7*7)+(64*5)+0], movePlane[(64*7*7)+(64*5)+1],
                                movePlane[(64*7*7)+(64*5)+8], movePlane[(64*7*7)+(64*5)+9],
                                movePlane[((64*7*7)+64*6):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*7)+64*4)], movePlane[(64*7*7)+(64*4)+0],
                                movePlane[(64*7*7)+(64*4)+1], movePlane[(64*7*7)+(64*4)+2],
                                movePlane[(64*7*7)+(64*4)+8], movePlane[(64*7*7)+(64*4)+9],
                                movePlane[(64*7*7)+(64*4)+10], movePlane[(64*7*7)+(64*4)+16],
                                movePlane[(64*7*7)+(64*4)+17], movePlane[(64*7*7)+(64*4)+18],
                                movePlane[((64*7*7)+64*5):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*7)+64*3)], movePlane[(64*7*7)+(64*3)+0:(64*7*7)+(64*3)+4],
                                movePlane[(64*7*7)+(64*3)+8:(64*7*7)+(64*3)+12], movePlane[(64*7*7)+(64*3)+16:(64*7*7)+(64*3)+20],
                                movePlane[(64*7*7)+(64*3)+24:(64*7*7)+(64*3)+28],
                                movePlane[((64*7*7)+64*4):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*7)+64*2)], movePlane[(64*7*7)+(64*2)+0:(64*7*7)+(64*2)+5],
                                movePlane[(64*7*7)+(64*2)+8:(64*7*7)+(64*2)+13], movePlane[(64*7*7)+(64*2)+16:(64*7*7)+(64*2)+21],
                                movePlane[(64*7*7)+(64*2)+24:(64*7*7)+(64*2)+29], movePlane[(64*7*7)+(64*2)+32:(64*7*7)+(64*2)+37],
                                movePlane[((64*7*7)+64*3):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*7)+64*1)], movePlane[(64*7*7)+(64*1)+0:(64*7*7)+(64*1)+6],
                                movePlane[(64*7*7)+(64*1)+8:(64*7*7)+(64*1)+14], movePlane[(64*7*7)+(64*1)+16:(64*7*7)+(64*1)+22],
                                movePlane[(64*7*7)+(64*1)+24:(64*7*7)+(64*1)+30], movePlane[(64*7*7)+(64*1)+32:(64*7*7)+(64*1)+38],
                                movePlane[(64*7*7)+(64*1)+40:(64*7*7)+(64*1)+46],
                                movePlane[((64*7*7)+64*2):]), axis=None)

    # remove unnecessary parts of west moves
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*6)+57], movePlane[((64*7*6)+64*6)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*6)+49], movePlane[((64*7*6)+64*6)+56:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*6)+41], movePlane[((64*7*6)+64*6)+48:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*6)+33], movePlane[((64*7*6)+64*6)+40:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*6)+25], movePlane[((64*7*6)+64*6)+32:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*6)+17], movePlane[((64*7*6)+64*6)+24:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*6)+9], movePlane[((64*7*6)+64*6)+16:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*6)+1], movePlane[((64*7*6)+64*6)+8:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*6)+64*5)+58], movePlane[((64*7*6)+64*5)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*5)+50], movePlane[((64*7*6)+64*5)+56:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*5)+42], movePlane[((64*7*6)+64*5)+48:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*5)+34], movePlane[((64*7*6)+64*5)+40:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*5)+26], movePlane[((64*7*6)+64*5)+32:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*5)+18], movePlane[((64*7*6)+64*5)+24:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*5)+10], movePlane[((64*7*6)+64*5)+16:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*5)+2], movePlane[((64*7*6)+64*5)+8:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*6)+64*4)+59], movePlane[((64*7*6)+64*4)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*4)+51], movePlane[((64*7*6)+64*4)+56:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*4)+43], movePlane[((64*7*6)+64*4)+48:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*4)+35], movePlane[((64*7*6)+64*4)+40:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*4)+27], movePlane[((64*7*6)+64*4)+32:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*4)+19], movePlane[((64*7*6)+64*4)+24:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*4)+11], movePlane[((64*7*6)+64*4)+16:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*4)+3], movePlane[((64*7*6)+64*4)+8:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*6)+64*3)+60], movePlane[((64*7*6)+64*3)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*3)+52], movePlane[((64*7*6)+64*3)+56:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*3)+44], movePlane[((64*7*6)+64*3)+48:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*3)+36], movePlane[((64*7*6)+64*3)+40:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*3)+28], movePlane[((64*7*6)+64*3)+32:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*3)+20], movePlane[((64*7*6)+64*3)+24:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*3)+12], movePlane[((64*7*6)+64*3)+16:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*3)+4], movePlane[((64*7*6)+64*3)+8:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*6)+64*2)+61], movePlane[((64*7*6)+64*2)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*2)+53], movePlane[((64*7*6)+64*2)+56:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*2)+45], movePlane[((64*7*6)+64*2)+48:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*2)+37], movePlane[((64*7*6)+64*2)+40:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*2)+29], movePlane[((64*7*6)+64*2)+32:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*2)+21], movePlane[((64*7*6)+64*2)+24:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*2)+13], movePlane[((64*7*6)+64*2)+16:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*2)+5], movePlane[((64*7*6)+64*2)+8:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*6)+64*1)+62], movePlane[((64*7*6)+64*1)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*1)+54], movePlane[((64*7*6)+64*1)+56:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*1)+46], movePlane[((64*7*6)+64*1)+48:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*1)+38], movePlane[((64*7*6)+64*1)+40:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*1)+30], movePlane[((64*7*6)+64*1)+32:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*1)+22], movePlane[((64*7*6)+64*1)+24:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*1)+14], movePlane[((64*7*6)+64*1)+16:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*1)+6], movePlane[((64*7*6)+64*1)+8:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*6)+64*0)+63], movePlane[((64*7*6)+64*0)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*0)+55], movePlane[((64*7*6)+64*0)+56:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*0)+47], movePlane[((64*7*6)+64*0)+48:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*0)+39], movePlane[((64*7*6)+64*0)+40:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*0)+31], movePlane[((64*7*6)+64*0)+32:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*0)+23], movePlane[((64*7*6)+64*0)+24:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*0)+15], movePlane[((64*7*6)+64*0)+16:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*6)+64*0)+7], movePlane[((64*7*6)+64*0)+8:]), axis=None)


    # keep SW corner
    movePlane = np.concatenate((movePlane[:((64*7*5)+64*6)], movePlane[(64*7*5)+64*6+56], movePlane[64*7*6:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*5)+64*5)], movePlane[(64*7*5)+(64*5)+48], movePlane[(64*7*5)+(64*5)+49],
                                movePlane[(64*7*5)+(64*5)+56], movePlane[(64*7*5)+(64*5)+57],
                                movePlane[((64*7*5)+64*6):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*5)+64*4)], movePlane[(64*7*5)+(64*4)+40],
                                movePlane[(64*7*5)+(64*4)+41], movePlane[(64*7*5)+(64*4)+42],
                                movePlane[(64*7*5)+(64*4)+48], movePlane[(64*7*5)+(64*4)+49],
                                movePlane[(64*7*5)+(64*4)+50], movePlane[(64*7*5)+(64*4)+56],
                                movePlane[(64*7*5)+(64*4)+57], movePlane[(64*7*5)+(64*4)+58],
                                movePlane[((64*7*5)+64*5):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*5)+64*3)], movePlane[(64*7*5)+(64*3)+32:(64*7*5)+(64*3)+36],
                                movePlane[(64*7*5)+(64*3)+40:(64*7*5)+(64*3)+44], movePlane[(64*7*5)+(64*3)+48:(64*7*5)+(64*3)+52],
                                movePlane[(64*7*5)+(64*3)+56:(64*7*5)+(64*3)+60],
                                movePlane[((64*7*5)+64*4):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*5)+64*2)], movePlane[(64*7*5)+(64*2)+24:(64*7*5)+(64*2)+29],
                                movePlane[(64*7*5)+(64*2)+32:(64*7*5)+(64*2)+37], movePlane[(64*7*5)+(64*2)+40:(64*7*5)+(64*2)+45],
                                movePlane[(64*7*5)+(64*2)+48:(64*7*5)+(64*2)+53], movePlane[(64*7*5)+(64*2)+56:(64*7*5)+(64*2)+61],
                                movePlane[((64*7*5)+64*3):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*5)+64*1)],
                                movePlane[(64*7*5)+(64*1)+16:(64*7*5)+(64*1)+22], movePlane[(64*7*5)+(64*1)+24:(64*7*5)+(64*1)+30],
                                movePlane[(64*7*5)+(64*1)+32:(64*7*5)+(64*1)+38], movePlane[(64*7*5)+(64*1)+40:(64*7*5)+(64*1)+46],
                                movePlane[(64*7*5)+(64*1)+48:(64*7*5)+(64*1)+54], movePlane[(64*7*5)+(64*1)+56:(64*7*5)+(64*1)+62],
                                movePlane[((64*7*5)+64*2):]), axis=None)

    # remove unnecessary parts of south moves
    movePlane = np.concatenate((movePlane[:((64*7*4)+64*6)], movePlane[((64*7*4)+64*6)+56:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*4)+64*5)], movePlane[((64*7*4)+64*5)+48:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*4)+64*4)], movePlane[((64*7*4)+64*4)+40:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*4)+64*3)], movePlane[((64*7*4)+64*3)+32:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*4)+64*2)], movePlane[((64*7*4)+64*2)+24:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*4)+64*1)], movePlane[((64*7*4)+64*1)+16:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*4)+64*0)], movePlane[((64*7*4)+64*0)+8:]), axis=None)

    # keep SE corner
    movePlane = np.concatenate((movePlane[:((64*7*3)+64*6)], movePlane[(64*7*4)-1:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*3)+64*5)], movePlane[(64*7*3)+(64*5)+54], movePlane[(64*7*3)+(64*5)+55],
                                movePlane[(64*7*3)+(64*5)+62], movePlane[(64*7*3)+(64*5)+63],
                                movePlane[((64*7*3)+64*6):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*3)+64*4)], movePlane[(64*7*3)+(64*4)+45],
                                movePlane[(64*7*3)+(64*4)+46], movePlane[(64*7*3)+(64*4)+47],
                                movePlane[(64*7*3)+(64*4)+53], movePlane[(64*7*3)+(64*4)+54],
                                movePlane[(64*7*3)+(64*4)+55], movePlane[(64*7*3)+(64*4)+61],
                                movePlane[(64*7*3)+(64*4)+62], movePlane[(64*7*3)+(64*4)+63],
                                movePlane[((64*7*3)+64*5):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*3)+64*3)], movePlane[(64*7*3)+(64*3)+36:(64*7*3)+(64*3)+40],
                                movePlane[(64*7*3)+(64*3)+44:(64*7*3)+(64*3)+48], movePlane[(64*7*3)+(64*3)+52:(64*7*3)+(64*3)+56],
                                movePlane[(64*7*3)+(64*3)+60:(64*7*3)+(64*3)+64],
                                movePlane[((64*7*3)+64*4):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*3)+64*2)], movePlane[(64*7*3)+(64*2)+27:(64*7*3)+(64*2)+32],
                                movePlane[(64*7*3)+(64*2)+35:(64*7*3)+(64*2)+40], movePlane[(64*7*3)+(64*2)+43:(64*7*3)+(64*2)+48],
                                movePlane[(64*7*3)+(64*2)+51:(64*7*3)+(64*2)+56], movePlane[(64*7*3)+(64*2)+59:(64*7*3)+(64*2)+64],
                                movePlane[((64*7*3)+64*3):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*3)+64*1)],
                                movePlane[(64*7*3)+(64*1)+18:(64*7*3)+(64*1)+24], movePlane[(64*7*3)+(64*1)+26:(64*7*3)+(64*1)+32],
                                movePlane[(64*7*3)+(64*1)+34:(64*7*3)+(64*1)+40], movePlane[(64*7*3)+(64*1)+42:(64*7*3)+(64*1)+48],
                                movePlane[(64*7*3)+(64*1)+50:(64*7*3)+(64*1)+56], movePlane[(64*7*3)+(64*1)+58:(64*7*3)+(64*1)+64],
                                movePlane[((64*7*3)+64*2):]), axis=None)

    # remove unnecessary parts of east moves
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*6)+56], movePlane[((64*7*2)+64*6)+63:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*6)+48], movePlane[((64*7*2)+64*6)+55:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*6)+40], movePlane[((64*7*2)+64*6)+47:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*6)+32], movePlane[((64*7*2)+64*6)+39:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*6)+24], movePlane[((64*7*2)+64*6)+31:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*6)+16], movePlane[((64*7*2)+64*6)+23:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*6)+8], movePlane[((64*7*2)+64*6)+15:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*6)+0], movePlane[((64*7*2)+64*6)+7:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*2)+64*5)+56], movePlane[((64*7*2)+64*5)+62:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*5)+48], movePlane[((64*7*2)+64*5)+54:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*5)+40], movePlane[((64*7*2)+64*5)+46:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*5)+32], movePlane[((64*7*2)+64*5)+38:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*5)+24], movePlane[((64*7*2)+64*5)+30:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*5)+16], movePlane[((64*7*2)+64*5)+22:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*5)+8], movePlane[((64*7*2)+64*5)+14:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*5)+0], movePlane[((64*7*2)+64*5)+6:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*2)+64*4)+56], movePlane[((64*7*2)+64*4)+61:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*4)+48], movePlane[((64*7*2)+64*4)+53:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*4)+40], movePlane[((64*7*2)+64*4)+45:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*4)+32], movePlane[((64*7*2)+64*4)+37:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*4)+24], movePlane[((64*7*2)+64*4)+29:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*4)+16], movePlane[((64*7*2)+64*4)+21:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*4)+8], movePlane[((64*7*2)+64*4)+13:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*4)+0], movePlane[((64*7*2)+64*4)+5:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*2)+64*3)+56], movePlane[((64*7*2)+64*3)+60:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*3)+48], movePlane[((64*7*2)+64*3)+52:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*3)+40], movePlane[((64*7*2)+64*3)+44:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*3)+32], movePlane[((64*7*2)+64*3)+36:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*3)+24], movePlane[((64*7*2)+64*3)+28:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*3)+16], movePlane[((64*7*2)+64*3)+20:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*3)+8], movePlane[((64*7*2)+64*3)+12:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*3)+0], movePlane[((64*7*2)+64*3)+4:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*2)+64*2)+56], movePlane[((64*7*2)+64*2)+59:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*2)+48], movePlane[((64*7*2)+64*2)+51:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*2)+40], movePlane[((64*7*2)+64*2)+43:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*2)+32], movePlane[((64*7*2)+64*2)+35:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*2)+24], movePlane[((64*7*2)+64*2)+27:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*2)+16], movePlane[((64*7*2)+64*2)+19:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*2)+8], movePlane[((64*7*2)+64*2)+11:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*2)+0], movePlane[((64*7*2)+64*2)+3:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*2)+64*1)+56], movePlane[((64*7*2)+64*1)+58:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*1)+48], movePlane[((64*7*2)+64*1)+50:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*1)+40], movePlane[((64*7*2)+64*1)+42:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*1)+32], movePlane[((64*7*2)+64*1)+34:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*1)+24], movePlane[((64*7*2)+64*1)+26:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*1)+16], movePlane[((64*7*2)+64*1)+18:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*1)+8], movePlane[((64*7*2)+64*1)+10:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*1)+0], movePlane[((64*7*2)+64*1)+2:]), axis=None)

    movePlane = np.concatenate((movePlane[:((64*7*2)+64*0)+56], movePlane[((64*7*2)+64*0)+57:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*0)+48], movePlane[((64*7*2)+64*0)+49:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*0)+40], movePlane[((64*7*2)+64*0)+41:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*0)+32], movePlane[((64*7*2)+64*0)+33:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*0)+24], movePlane[((64*7*2)+64*0)+25:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*0)+16], movePlane[((64*7*2)+64*0)+17:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*0)+8], movePlane[((64*7*2)+64*0)+9:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*2)+64*0)+0], movePlane[((64*7*2)+64*0)+1:]), axis=None)

    # keep NE corner
    movePlane = np.concatenate((movePlane[:((64*7*1)+64*6)], movePlane[(64*7*1)+64*6+7], movePlane[64*7*2:]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*1)+64*5)], movePlane[(64*7*1)+(64*5)+6], movePlane[(64*7*1)+(64*5)+7],
                                movePlane[(64*7*1)+(64*5)+14], movePlane[(64*7*1)+(64*5)+15],
                                movePlane[((64*7*1)+64*6):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*1)+64*4)], movePlane[(64*7*1)+(64*4)+5],
                                movePlane[(64*7*1)+(64*4)+6], movePlane[(64*7*1)+(64*4)+7],
                                movePlane[(64*7*1)+(64*4)+13], movePlane[(64*7*1)+(64*4)+14],
                                movePlane[(64*7*1)+(64*4)+15], movePlane[(64*7*1)+(64*4)+21],
                                movePlane[(64*7*1)+(64*4)+22], movePlane[(64*7*1)+(64*4)+23],
                                movePlane[((64*7*1)+64*5):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*1)+64*3)], movePlane[(64*7*1)+(64*3)+4:(64*7*1)+(64*3)+8],
                                movePlane[(64*7*1)+(64*3)+12:(64*7*1)+(64*3)+16], movePlane[(64*7*1)+(64*3)+20:(64*7*1)+(64*3)+24],
                                movePlane[(64*7*1)+(64*3)+28:(64*7*1)+(64*3)+32],
                                movePlane[((64*7*1)+64*4):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*1)+64*2)], movePlane[(64*7*1)+(64*2)+3:(64*7*1)+(64*2)+8],
                                movePlane[(64*7*1)+(64*2)+11:(64*7*1)+(64*2)+16], movePlane[(64*7*1)+(64*2)+19:(64*7*1)+(64*2)+24],
                                movePlane[(64*7*1)+(64*2)+27:(64*7*1)+(64*2)+32], movePlane[(64*7*1)+(64*2)+35:(64*7*1)+(64*2)+40],
                                movePlane[((64*7*1)+64*3):]), axis=None)
    movePlane = np.concatenate((movePlane[:((64*7*1)+64*1)],
                                movePlane[(64*7*1)+(64*1)+2:(64*7*1)+(64*1)+8], movePlane[(64*7*1)+(64*1)+10:(64*7*1)+(64*1)+16],
                                movePlane[(64*7*1)+(64*1)+18:(64*7*1)+(64*1)+24], movePlane[(64*7*1)+(64*1)+26:(64*7*1)+(64*1)+32],
                                movePlane[(64*7*1)+(64*1)+34:(64*7*1)+(64*1)+40], movePlane[(64*7*1)+(64*1)+42:(64*7*1)+(64*1)+48],
                                movePlane[((64*7*1)+64*2):]), axis=None)

    # remove unnecessary parts of north moves
    movePlane = np.concatenate((movePlane[:(64*6)+8], movePlane[(64*6)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:(64*5)+16], movePlane[(64*5)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:(64*4)+24], movePlane[(64*4)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:(64*3)+32], movePlane[(64*3)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:(64*2)+40], movePlane[(64*2)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:(64*1)+48], movePlane[(64*1)+64:]), axis=None)
    movePlane = np.concatenate((movePlane[:(64*0)+56], movePlane[(64*0)+64:]), axis=None)

    knightMovePlane = knightMovePlane.flatten()
    # direction ([1, 2],[2, 1],[2, -1],[1, -2],[-1, -2],[-2, -1],[-2, 1],[-1, 2])

    # removing unnecessary north/south squares for knight moves
    knightMovePlane = knightMovePlane[0:(64*7)+48]
    knightMovePlane = np.concatenate((knightMovePlane[:(64*6)+56], knightMovePlane[(64*6)+64:]), axis=None)
    knightMovePlane = np.concatenate((knightMovePlane[:(64*5)], knightMovePlane[(64*5)+8:]), axis=None)
    knightMovePlane = np.concatenate((knightMovePlane[:(64*4)], knightMovePlane[(64*4)+16:]), axis=None)
    knightMovePlane = np.concatenate((knightMovePlane[:(64*3)], knightMovePlane[(64*3)+16:]), axis=None)
    knightMovePlane = np.concatenate((knightMovePlane[:(64*2)], knightMovePlane[(64*2)+8:]), axis=None)
    knightMovePlane = np.concatenate((knightMovePlane[:(64*1)+56], knightMovePlane[(64*1)+64:]), axis=None)
    knightMovePlane = np.concatenate((knightMovePlane[:(64*0)+48], knightMovePlane[(64*0)+64:]), axis=None)

    # removing unnecessary east/west squares for knight moves

    underPromotion = underPromotion.flatten()
    moveToArray = np.concatenate((placedPlane, movePlane, knightMovePlane, underPromotion))
    #end = time.time()
    #print(end-start)
    if np.sum(moveToArray) != 1:
        print("error")
    return moveToArray.reshape((1, 2308))

def placementPieceAvailable(captivePieces):
    newArray = [0, 0, 0, 0, 0]
    for i in range(5):
        if captivePieces[i] > 0:
            newArray[i] = 1
    return newArray

def moveEvaluation(move, board, prediction):

    prediction = prediction.detach().numpy().flatten()
    prediction = np.exp(prediction)
    move = moveArray(move, board).flatten()
    if np.sum(move) != 1:
        print("error")
    evaluate = (move * prediction)
    finalEval = np.sum(evaluate)

    return finalEval

def moveEvaluations(legalMoves, board, prediction):
    evals = np.zeros((len(legalMoves)))
    for i in range(len(evals)):
        evals[i] = moveEvaluation(legalMoves[i], board, prediction)
    return evals

def sortEvals(moveNames, scores):

    for i in range(len(moveNames)):
        for j in range(len(moveNames)):
            if scores[j] < scores[i]:
                temp = moveNames[i]
                moveNames[i] = moveNames[j]
                moveNames[j] = temp

                temp = scores[i]
                scores[i] = scores[j]
                scores[j] = temp

    return moveNames

def boardToInt(state):
    state = state.flatten()
    string = "0"*960
    for i in range(len(state)):
        if state[i] == 1:
            string = string[0:i]+"1"+string[i+1:960]
    return string

# Instead of boardToInt which saves the board into a string of 960 0s and 1s,
# this attempts to save the board into an array of 15 values.
def boardToBinaryArray(state):
    board = np.zeros(15)
    state = state.flatten()
    string = "0"*960
    for i in range(len(state)):
        if state[i] == 1:
            string = string[0:i]+"1"+string[i+1:960]
    for j in range(15):
        binaryPlaneString = string[64*j:64*j+64]
        binaryPlaneNumber = int(binaryPlaneString, 2)
        board[j] = binaryPlaneNumber
    return board

def binaryArrayToBoard(array):
    string = ""
    for i in range(15): #array should be of length 15
        string = string + ("{:064b}".format(int(array[i])))
    inArray = np.zeros(960)
    for i in range(960):
        inArray[i] = int(string[i])
    inArray = np.reshape(inArray, (15, 8, 8))
    return inArray





testing = False
if testing:

    board = ChessEnvironment()
    board.makeMove("g1f3")
    board.makeMove("g8f6")
    board.makeMove("f3g1")
    board.makeMove("f6g8")
    print(board.board.is_repetition(count=2))



