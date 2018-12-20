"""
This python file just generates new training games with noise, for the computer to learn from.

Protocol:

createTrainingGames( number of games you want, number of playouts after each position )
"""

import numpy as np
from MCTSCrazyhouse import MCTS
from EnsembleMCTSCrazyhouse import EnsembleMCTS

treeSearch = MCTS('New Networks/18011810-POLICY.pt', 'New Networks/18011810-VALUE.pt', 1)
treeSearch.createTrainingGames(20, 0, 'Self-Play Games/Games1to20-PLAYOUTS0-DEPTH0.pgn')
