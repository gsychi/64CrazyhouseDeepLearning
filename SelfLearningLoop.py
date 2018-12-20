"""
This is a loop that keeps training the engine on self-play data.

"""
import numpy as np
from MCTSCrazyhouse import MCTS
import TrainPolicyNetwork
import TrainValueNetwork
import NNArenaCompetition
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
LEARNING_RATE = 0.001

"""
STEP 0: DETERMINE A STARTING POINT:
"""
BEST_POLICY = 'New Networks/18011810-POLICY.pt'
BEST_VALUE = 'New Networks/18011810-VALUE.pt'


# 100 generations
for i in range(GENERATIONS):

    BEST_NETWORK = MCTS(BEST_POLICY, BEST_VALUE, TRAINING_DEPTH)

    """
    STEP 1: GENERATE TRAINING GAMES!
    """
    print("Best Network:", BEST_NETWORK.nameOfNetwork)
    saveDir = "Self-Play Games/" + BEST_NETWORK.nameOfNetwork + "-DEPTH" + TRAINING_PLAYOUTS + "-PLAYOUTS" + BEST_NETWORK.DEPTH_VALUE + ".pgn"
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
    randomHexP = secrets.token_hex(5)+"-POLICY.pt"
    randomHexV = secrets.token_hex(5)+"-POLICY.pt"
    print(randomHexP)
    print("Start Training.")
    # TRAIN A POLICY AND A VALUE NETWORK
    # TRAIN TRAIN TRAIN...

    """
    STEP FOUR: TEST NEW NETWORKS:
    """
    TEST_NETWORK = MCTS(randomHexP, randomHexV, TESTING_DEPTH)
    BEST_NETWORK = BEST_NETWORK = MCTS(BEST_POLICY, BEST_VALUE, TESTING_DEPTH)
    replace = NNArenaCompetition.bestNetworkTest(BEST_NETWORK, TEST_NETWORK, TESTING_GAMES, TESTING_PLAYOUTS)
    if replace:
        BEST_POLICY = randomHexP
        BEST_VALUE = randomHexV




