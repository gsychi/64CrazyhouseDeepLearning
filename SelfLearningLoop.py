"""
This is a loop that keeps training the engine on self-play data.

"""
import numpy as np
from MCTSCrazyhouse import MCTS
import ServerTraining
import NNArenaCompetition
import secrets

# start off by choosing a neural network directory.
BEST_NETWORK = MCTS('supervised2.pt')
BEST_NETWORK_DIRECTORY = BEST_NETWORK.nameOfNetwork + ".pt"

# parameters
GENERATIONS = 100
TRAINING_GAMES = 50
TESTING_GAMES = 10
TRAINING_PLAYOUTS = 1
TESTING_PLAYOUTS = 2

EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 100 generations
for i in range(GENERATIONS):

    print("Best Network:", BEST_NETWORK_DIRECTORY)

    # first, create self trained games.
    selfPlayInput, selfPlayOutput = BEST_NETWORK.createTrainingGames(TRAINING_GAMES, TRAINING_PLAYOUTS)
    print("Training games generated. On to training.")
    # selfPlayInput = np.load("masterInputs.npy")
    # selfPlayOutput = np.load("masterOutputs.npy")

    if len(selfPlayInput)/BATCH_SIZE < 40:
        BATCH_SIZE = len(selfPlayInput)/50

    # train your neural network on this material.
    randomHex = secrets.token_hex(5)+".pt"
    print(randomHex)
    print("Start Training.")
    ServerTraining.trainNetwork(selfPlayInput, selfPlayOutput, loadDirectory="BEST_NETWORK_DIRECTORY", saveDirectory=randomHex,  # see if we should train from scratch or not...
                                EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, updateInterval=2, LR=LEARNING_RATE)


    TEST_NETWORK = MCTS(randomHex)
    # Now that a new network is trained, we get the new network to compete against the past one.
    # Before starting the competition, make sure that the best network has all its previous searches cleared.
    BEST_NETWORK.clearInformation()

    replace = NNArenaCompetition.bestNetworkTest(BEST_NETWORK, TEST_NETWORK, TESTING_GAMES, TESTING_PLAYOUTS)
    if replace:
        BEST_NETWORK_DIRECTORY = randomHex
        BEST_NETWORK = MCTS(BEST_NETWORK_DIRECTORY)

    # now all the information is updated. Generation is done.
