"""
This python file just generates new training games with noise, for the computer to learn from.

Protocol:

createTrainingGames( number of games you want, number of playouts after each position, save directory)
"""

import numpy as np
from MCTSCrazyhouse import MCTS

treeSearch = MCTS('New Networks/(MCTS)(6X128|4|8)(V1)64fish.pt', 0)
treeSearch.createTrainingGames(20, 0, 'Self-Play Games/Games1to20-PLAYOUTS0-DEPTH0.pgn')
