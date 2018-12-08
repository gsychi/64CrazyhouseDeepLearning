"""
This is quite an important function, as it maps each action that the computer chooses
"""
import numpy as np
import chess.variant
import copy
from scipy.interpolate import interp1d

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

# move is a string, turns into array. pick-up square should have value 2 instead of 1.
def moveArray(move, board, notArrayToString=True):
    placedPlane = np.zeros((5, 8, 8))  # pawn, knight, bishop, rook, queen.
    pickUpPlane = np.zeros((8, 8))
    movePlane = np.zeros((8, 7, 8, 8))  # direction (N, NE, E, SE, S, SW, W, NW), squares.
    knightMovePlane = np.zeros((8, 8, 8))  # direction ([1, 2],[2, 1],[2, -1],[1, -2],[-1, -2],[-2, -1],[-2, 1],[-1, 2])
    underPromotion = np.zeros((3, 8))  # this can be a 8x8 plane, but for now we will not. Knight, Bishop, Rook

    if "PRNBQ".find(move[0]) == -1:
        #print("MOVE TO ARRAY: NORMAL MOVE")
        rowNames = "abcdefgh"
        if notArrayToString:
            pickUpPlane[8 - int(move[1])][int(rowNames.find(move[0]))] = 1

        # identify how far the piece has moved, and how far it will be moving.
        if "PBRQK".find(board[8 - int(move[1])][int(rowNames.find(move[0]))].upper()) != -1:
            # print("its a", board[8 - int(move[1])][int(rowNames.find(move[0]))])
            if len(move) == 5:
                directory = "nbr".find(move[4].lower())  # .lower() just in case
                if directory != -1:
                    underPromotion[directory][int(rowNames.find(move[2]))] = 1
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
            rowMovement = int(move[3]) - int(move[1])  # positive = north, negative = south [NORTH = 0, SOUTH = 4]
            columnMovement = int(rowNames.find(move[2])) - int(
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

    # return the result.
    placedPlane = placedPlane.flatten()
    pickUpPlane = pickUpPlane.flatten()
    movePlane = movePlane.flatten()
    knightMovePlane = knightMovePlane.flatten()
    underPromotion = underPromotion.flatten()
    moveToArray = np.concatenate((placedPlane, pickUpPlane, movePlane, knightMovePlane, underPromotion))
    # ARRAY IS 4504 ENTRIES LONG
    return moveToArray.reshape((1, 4504))

def placementPieceAvailable(captivePieces):
    newArray = [0, 0, 0, 0, 0]
    for i in range(5):
        if captivePieces[i] > 0:
            newArray[i] = 1
    return newArray

# turns array into string! yay
def moveArrayToString(array, board, pythonChessBoard, whiteCaptivePieces, blackCaptivePieces, plies):

    # extract individual planes, and then reshape them.
    placedPlane = array[0, 0:64 * 5].reshape((5, 8, 8))
    pickUpPlane = array[0, 64 * 5:(64 * 5) + (64 * 1)].reshape((8, 8))
    movePlane = array[0, 64 * 6:(64 * 6) + (8 * 7 * 8 * 8)].reshape((8, 7, 8, 8))
    knightMovePlane = array[0, (64 * 6) + (8 * 7 * 8 * 8):(64 * 6) + (8 * 7 * 8 * 8) + (8 * 8 * 8)].reshape((8, 8, 8))
    underPromotion = array[0, -24:].reshape(3, 8)

    # Start off by looking at the placedPlane and pickUpPlane and find out what is the highest value. For
    # placedPlane, we look at all legal places to drop each piece for each plane, and find out what is highest.
    # For pickUpPlane, we look at all the places in which there is a piece, and find out which piece has the highest
    # pick up score.

    wherePiecesAre = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if board[i][j] == " ":
                wherePiecesAre[i][j] = 0
            else:
                wherePiecesAre[i][j] = 1
    canPlace = 1 - wherePiecesAre
    pawnLegality = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == 0 or i == 7:
                pawnLegality[i][j] = 0
            else:
                pawnLegality[i][j] = 1

    pawnCanPlace = (canPlace + pawnLegality) - 1
    # +1 OCCURS ONLY WHEN cannotPlace and pawnLegality both output value 1. 0 if it's illegal.
    for i in range(8):
        for j in range(8):
            if pawnCanPlace[i][j] == -1:
                pawnCanPlace[i][j] = 0

    whiteCaptivePieces = placementPieceAvailable(whiteCaptivePieces)
    blackCaptivePieces = placementPieceAvailable(blackCaptivePieces)

    if plies % 2 == 0:
        placingLegal = np.concatenate(((pawnCanPlace.flatten() * whiteCaptivePieces[0]),
                                       (canPlace.flatten() * whiteCaptivePieces[1]),
                                       (canPlace.flatten() * whiteCaptivePieces[2]),
                                       (canPlace.flatten() * whiteCaptivePieces[3]),
                                       (canPlace.flatten() * whiteCaptivePieces[4])
                                       ))
    else:
        placingLegal = np.concatenate(((pawnCanPlace.flatten() * blackCaptivePieces[0]),
                                       (canPlace.flatten() * blackCaptivePieces[1]),
                                       (canPlace.flatten() * blackCaptivePieces[2]),
                                       (canPlace.flatten() * blackCaptivePieces[3]),
                                       (canPlace.flatten() * blackCaptivePieces[4])
                                       ))

    #need to use deep copy
    illegalPenalty = copy.deepcopy(placingLegal)
    for i in range(len(illegalPenalty)):
        if illegalPenalty[i] == 0:
            illegalPenalty[i] = -1000000
        else:
            illegalPenalty[i] = 0


    #print("Highest Placed Value:")
    directory1 = np.argmax(((array[0, 0:64*5]*placingLegal)+illegalPenalty).reshape(1, 64*5), axis=1)[0]
    maxPlaced = ((array[0, 0:64*5]*placingLegal)+illegalPenalty).reshape(1, 64*5)[0, directory1]
    #print(maxPlaced)

    # Computer must pick up a piece that can move.
    #print("Highest Pick-Up Piece Value:")
    directory2 = np.argmax((array[0, 64*5:64*6]*moveablePieces(board, pythonChessBoard).flatten()).reshape(1, 64)
                           + moveablePiecesPenalty(board, pythonChessBoard).flatten(), axis=1)[0]
    maxPickUp = pickUpPlane[directory2//8][directory2 % 8]
    #print(maxPickUp)



    # identify the square, the action, and the piece that is being moved.
    rowNames = "abcdefgh"

    if maxPlaced > maxPickUp:
        #print("NN evaluates placement of piece over normal move.")
        possiblePieces = "PNBRQ"
        piecePlaced = possiblePieces[directory1 // 64]
        initialRow = rowNames[directory1 % 8]
        initialCol = str(8 - ((directory1 % 64) // 8))
        move = (piecePlaced + "@" + initialRow + initialCol)
        #print("MOVE:")
        #print(move)

        # move this above maxPickUp > maxPlaced.
        # there needs to be a check for legality...and if it is unfulfilled, then
        # pickup another piece.
        if chess.Move.from_uci(move) in pythonChessBoard.legal_moves:
            return move
        else:
            # should just play a random legal move.
            maxPlaced = -1
            maxPickUp = 1

    if maxPickUp >= maxPlaced:
        #print("NN evaluates normal move over placement of piece.")

        # now, we return a list of legal moves.
        searchPossibilities = legalMovesFromSquare(directory2, board, pythonChessBoard)
        #print(len(searchPossibilities))
        #print("Possible Moves: ", searchPossibilities)
        #print(pythonChessBoard.legal_moves)

        # Then, find where the move should be on the array, and argmax all the possible moves.

        if len(searchPossibilities) > 0:
            bestMove = searchPossibilities[0]    # need to get the score for the move as well
            bestMoveScore = array[0][np.argmax(moveArray(searchPossibilities[0], board, False))]
            #print(bestMoveScore)
            if len(searchPossibilities) > 1:
                for i in range(1, len(searchPossibilities)):
                    # find a better move...?
                    newSearch = searchPossibilities[i]
                    newSearchScore = array[0][np.argmax(moveArray(searchPossibilities[i], board, False))]
                    #print(newSearchScore)
                    if newSearchScore > bestMoveScore:
                        bestMove = newSearch
                        bestMoveScore = newSearchScore
        else:
            return '0000'
        return bestMove

def moveEvaluation(move, board, prediction):

    prediction = prediction.numpy().flatten()
    prediction = np.exp(prediction)
    move = moveArray(move, board).flatten()
    for i in range(320, 384):
        move[i] = 0
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


testing = False
if testing:
    testingBoard = [["r", "n", "b", "q", "k", "b", "n", "r"],  # 0 - 7
                    ["p", "p", "p", "p", " ", "p", "p", "p"],  # 8 - 15
                    [" ", " ", " ", " ", " ", " ", " ", " "],  # 16 - 23
                    [" ", " ", " ", " ", "p", " ", " ", " "],  # 24 - 31
                    [" ", " ", " ", " ", "P", " ", " ", " "],  # 32 - 39
                    [" ", " ", " ", " ", " ", "Q", " ", " "],  # 40 - 47
                    ["P", "P", "P", "P", " ", "P", "P", "P"],  # 48 - 55
                    ["R", "N", "B", " ", "K", "B", "N", "R"]]  # 56 - 63

    whiteCaptivePieces = [0, 0, 0, 0, 0]
    blackCaptivePieces = [0, 0, 0, 0, 0]

    pythonBoard = chess.variant.CrazyhouseBoard()
    pythonBoard.push(chess.Move.from_uci("e2e4"))
    pythonBoard.push(chess.Move.from_uci("e7e5"))
    pythonBoard.push(chess.Move.from_uci("d1f3"))
    plies = 3

    A_MOVE = moveArray("f8a3", testingBoard)
    #print(A_MOVE)
