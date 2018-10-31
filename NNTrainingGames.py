"""
This python file just generates new training games with noise, for the computer to learn from.

Protocol:

createTrainingGames( number of games you want, number of playouts after each position )
"""

import numpy as np
from MCTSCrazyhouse import MCTS
from EnsembleMCTSCrazyhouse import EnsembleMCTS

treeSearch = EnsembleMCTS()
#treeSearch = MCTS('7 Layer k=32 Models/experimental.pt')
selfPlayInput, selfPlayOutput = treeSearch.createTrainingGames(200, 0)
np.save("selfPlay01Input.npy", selfPlayInput)
np.save("selfPlay01Output.npy", selfPlayOutput)
