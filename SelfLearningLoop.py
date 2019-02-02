"""
This is a loop that keeps training the engine on self-play data.

"""
import numpy as np
from MCTSCrazyhouse import MCTS
import secrets

# parameters
GENERATIONS = 100
TRAINING_GAMES = 1000
TESTING_GAMES = 250
TRAINING_PLAYOUTS = 0
TESTING_PLAYOUTS = 0
TRAINING_DEPTH = 1
TESTING_DEPTH = 1

EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.005

"""
STEP 0: DETERMINE A STARTING POINT:
"""
BEST_NET = "New Networks/[MCTS][6X128|4|8][V1]64fish.pt"

# 100 generations
for i in range(GENERATIONS):

    BEST_NETWORK = MCTS(BEST_NET, TRAINING_DEPTH)

    """
    STEP 1: GENERATE TRAINING GAMES!
    """
    print("Best Network:", BEST_NETWORK.nameOfNetwork)
    saveDir = "Self-Play Games/" + BEST_NETWORK.nameOfNetwork[13:] + "-DEPTH" + str(TRAINING_PLAYOUTS) + "-PLAYOUTS" + str(BEST_NETWORK.DEPTH_VALUE) + ".pgn"
    BEST_NETWORK.createTrainingGames(TRAINING_GAMES, TRAINING_PLAYOUTS, saveDir)
    print("Training games generated. On to training.")

    """
    STEP 2: CREATE TRAINING DATA
    """
    # TO BE DONE:
    # Use CreateFullDatabase.py
    selfPlayInput = []
    selfPlayOutput = []

    """
    STEP 3: TRAIN POLICY AND VALUE NETWORK
    """
    randomHex = "New Networks/" + secrets.token_hex(5) + ".pt"
    print(randomHex)
    print("Start Training.")
    # TRAIN TRAIN TRAIN...

    """
    STEP FOUR: TEST NEW NETWORKS:
    """




