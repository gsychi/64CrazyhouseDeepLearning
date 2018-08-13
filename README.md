# 64CrazyhouseDeepLearning
A deep learning Crazyhouse chess program that uses a Monte Carlo Tree Search (MCTS) based evaluation system and reinforcement to enhance its play style.

INTRODUCTION
In 2017, DeepMind released a generalized self-learning model of AlphaZero for Chess and Shogi that reached superhuman strength without any human knowledge. This project aims to implement a similar learning process for the chess variant of Crazyhouse, and at the same time, explore network architectures and search algorithms that make such programs successful.

There are a few reasons why the chess variant of Crazyhouse may pose a significant challenge to programmers. First, the game is still developing in terms of popularity and engine development. There are many normal-looking positions in which even the top Crazyhouse engine, Stockfish, may evaluate wrongly. While it has improved tremendously over the past few years, other engines are still far behind in terms of playing strength. Moreover, since theory is not as developed, hand-crafted features cannot merely rely on the countless theory knowledge that humans already possess. For this reason, it would be fascinating to potentially discover novelties or new principles.

Second, the approach of using depth-based search and neural networks seems sensical for this chess variant. While the action representation is not as intuitive as Go (the action array for this program is constructed by flattening multiple move planes), the variant has, on average, more legal moves per position, and is characterized by its large emphasis on more abstract ideas. Ideas such as gaining the initiative, exposing king safety, and saving piece drops seem difficult to be crafted by an evaluation function, but are ideas that programs like AlphaZero and Leela Zero show in the game of shogi and chess respectively. Additionally, since Leela Zero is known for being tactically weak but positionally strong, it would be interesting to see how a deep-learning based engine could ward off the many more tactical threats in Crazyhouse.

GOAL
The average Crazyhouse player on lichess.org has a rating of roughly 1600. For bots, these ratings are deflated, so I may consider asking for permission to run a normal account.

The first goal is for the engine to first attain the level of an amateur (~1600) player at Crazyhouse through supervised learning (training the engine on games played by masters) and, only if needed, reinforcement learning (have the trained network generate new games and play itself to improve its strength). If the decision is to proceed with RL, this will be where most of the energy will be spent for the next few months due to limitations in computational power and other testing requirements.

The second goal is for the engine to climb up to the Top 200 list in Crazyhouse (2000+) on lichess.org. This will use the same approach as previously mentioned, but with a (hopefully) deeper network, better training information, and self-training time.

Once this can all be achieved, and the code is perfected, we may finally try to start a self-learning loop similar to Leela Zero and its chess equivalent. From there, there will be many options available - we can pursue computer engine competitions or attempt to find new theory.

PROJECT STRUCTURE
Supervised + Reinforcement Learning

Similar to AlphaGo, we train a neural network which takes the raw board representation as the input, and from there, returns an output array that can be processed to find its evaluation of all legal moves from the position. The network is first trained from master games and return a probability of a move being played from a certain position.

We utilize a depth-first search UCT Algorithm from MCTS that 'intuitively' selects a move from the neural network evaluations, and plays against itself to the end. It then makes a decision based on its previous experience and the neural network.

To improve the neural network, we have the best neural network play against itself multiple times to generate more 'master' games for itself to learn from. New networks will then learn to predict the probability of winning given a move in a certain position, before competing against the previous best network. If the previous best network is defeated, then this newly trained network replaces the old one and generates more training data. This process continues for as long as possible.

NOTE: The program is designed such that it can start from completely random weights. However, this takes a long time to train and I am unsure about the resources that will/can be allocated to this project.

TRAINING INPUT
The input for the neural network is a basic representation of the board. It is an array of 896 values (14 planes of 8x8 boards), which determines the player turn, position of each piece type, captive pieces held by a player, and whether the piece is a promotion. The last part is relevant since promoted pieces can only be placed down as pawns after being captured.

TRAINING OUTPUT
The output for the neural network is a 1D representation of multiple move planes (8x8 boards). Here they are as followed:

The first five planes represent the possible squares in which a given piece should be placed (pawn, knight, bishop, rook, queen).

The next plane represents the piece that should be picked up from the board.

The next 56 planes represent the possible queen moves. There are 8 planes for each direction (N, NE, E, SE, S, SW, W, NW), and 7 planes for how far a piece can move (1-7). This is used to map all moves made by a pawn, bishop, rook, queen, and king.

The next 8 planes represent the possible knight moves (2F1L, 2F1R, 1F2L, 1F2R, 1B2L, 1B2R, 2B1L, 2B1R). This is used to map moves made by a knight.

Last, we have 3 planes of 8x1 boards, which determine the column in which a pawn is being underpromoted. Each plane denotes a different promotion (rook, bishop, knight).

In total, the action is represented by an array of (5+1+56+8) x (8x8) + 3x8 = 4504 values.

NEURAL NETWORK ARCHITECTURE

This project has two neural networks ready for testing.

The first uses a convolutional neural network from the PyTorch module. Right now, due to the computational power of my computer, the network consists of three layers and one fully connected layer. Each layer has a ReLu activation function and batch normalization.

First Layer: 8 filters, kernel size 5x5
Second Layer: 8 filters, kernel size 5x5
(Third Layer: 8 filters, kernel size 3x3)
Final Layer: Linear Layer
(NOTE: The final architecture for training for this project is not yet set. However, I aim to have at least 64 to 128 filters on each layer, and to have at least 6-8 layers on the convolutional neural network.)

The second is a flexible residual network framework where the user can specify the depth and architecture of each block. As of now, there are four template networks as introduced by He et al. : ResNet18, ResNet34, ResNet101, and ResNet152. The residual network has, due to its more demanding computational time, not been tested for the uses of the project yet.

At the current moment, each neural network has a final linear layer without an activation function, meaning that the network can return any real number (usually, the training data has a max value of roughly 10 [this will be explained later] and a minimum value of -30.)

Theoretically, this should be acceptable for supervised learning, where our problem is designed as a classification problem and the neural network only needs to identify better moves from worse ones. However, as we move towards using the MCTS tree search and starting the reinforcement learning process, it is imperative that the neural network only returns values from 0 to 1, as the Monte Carlo Tree Search (MCTS) reacts to the neural network evaluations as a probability of winning given a certain move in a position.

[It is theoretically possible to work around this by training the neural network such that it outputs the logit values of the win probability.

All outputs can be scaled by the function logit(x), where x denotes the probability of winning from 0 to 1. Since logit(x) is undefined at x = 1 and x = 0, we divide all probabilities of winning by 1.0002 and add 0.0001 to remove this error without our training data being compensated. In our master dataset, the frequency of a move given a position are scaled from 0 to 1 such that the main line gets a perfect score of 1, before using the above manipulation.

While this is theoretically sound, the neural network cannot find any patterns when the outputs are shaped as followed. Thus, for the master game dataset, we accept the problem and multiple the probability values by a factor of 20 for now.]

CODE BREAKDOWN
Since there are many files on the Github repository for this project that are not optimally labelled, this manual will provide a quick analysis of all the functions in the project thus far. I promise to write more annotations and a more in-depth analysis here.

ActionToArray.py - This file contains functions that convert move strings (i.e. 'e2e4') into the 4504 value-long output array, and similarly, convert the neural network output into one single move string. More importantly, there is a function that returns the sigmoid output from the neural network for each legal move in a position, and sorts this in an array that can be used by the Monte Carlo Tree Search class.

ChessConvNet.py / ChessResNet.py - These two files contain, as the name suggests, classes for the convolutional and residual network used in the project. The residual network has not been tested yet.

ChessEnvironment.py - This is the environment for Crazyhouse Chess that the neural networks will be playing on. Users can specify for the computer to read PGN files only when both players pass a certain Elo rating.

CreateDatabase.py - This file takes games downloaded from the lichess database and creates masterInputs and masterOutputs numpy files for the supervised learning.

MCTSCrazyhouse.py - This file contains a Monte Carlo Tree Search class that requires a neural network to run. It is capable of creating training games (neural network evaluations are affected by noise constants) and competitive games (no noise) and export the PGN file of each game. For training games, users specify the number of self-play games, from which this class returns selfPlayInput and selfPlayOutput numpy files. The selfPlayOutput is a matrix that trains a value network (each row is a win probability of seen moves.)

MyDataset.py - This file allows Pytorch to use a custom dataset.

NNArenaCompetition.py - This allows two neural networks to face off.

NNTrainingGames.py - This allows a neural network to generate self-play games.

SelfLearningLoop.py - This allows a user to declare the starting 'best' neural network, have it generate self-play games, from which the data is used to train a new network to replace the old 'best' neural network. It is a combination of NNArenaCompetition.py and NNTrainingGames.py

ServerTraining.py - This takes in the inputs and outputs saved from CreateDatabase.py

testing.py is irrelevant since it is never referenced in the program.

MAIN PRACTICAL ERRORS
1. The main problem, as mentioned previously, is the fact that the neural network does not output values from 0 to 1. The attempt to use a logit function confuses the computer, so it is essential to find another way to solve this solution. This must be solved.

2. CreateDatabase.py has a slow run time, and if anyone has a better method, I would be happy to edit or replace it with your work.

3. Almost all training games and competition games are resulting in draws.

POSSIBLE REASON: Previously, all self-training games were decisive, but this was because I did not use a sigmoid function to map the probabilities of a move. Rather, I used a linear function that took the minimum and maximum outputs by the neural network (-66 to 8) and set those values as 0 and 1 respectively; all values were scaled to fit this assumption.

With the sigmoid function, the computer evaluates all moves with less than 1e-03 probability of being played when reaching unknown positions out of the opening; thus, its conversion of winning position becomes much worse as the noise becomes much more decisive (noise around the 50th move should be around 1e-02, which doesn't affect the linearly-scaled evaluations).

I am hesitant to change back, since I want to eventually use win probabilities to limit the search depth of a playout. I suspect part of the problem is just due to the limited training and tests I can run with my laptop, which prevents me from training networks on big datasets and using more than 5 playouts.

If this problem persists, an algorithmic concession may be the only solution.

4. Constants for the playouts, i.e. noise and exploration factor, are completely arbitrary. Testing and changes will have to be made eventually, since the MCTS is a very important part in improving the intuitive suggestions by the neural network.

REQUIRED ADDITIONAL CODE 
Neural Network should only be able to output values from 0 to 1, and must be able achieve >80-85% accuracy on the supervised information. (Urgency: high)
Right now, all playouts last until the end of the game. However, in the future, the computer should be able to specify a search depth and use np.amax to determine the win probability in the final searched position (1-np.amax if depth is even). A playout should therefore be able to return any value from 0 to 1. (Urgency: medium)
Create a Leela-like online server training system for everyone to work with and to lend their GPUs on. (Urgency: medium)
Time Control - If this engine is to compete in competitions, it must be able to determine when it can search further, and whether it has the time to do so.(Urgency: low)
Translate into C++? (Urgency: low)
HOW CAN I CONTRIBUTE?
There are many ways to contribute to the project!

Since I am open-sourcing this project, I hope that there will be people willing to train networks and test the success of the constants. Similarly, everyone is welcome to improve upon the code and solve any of the outlined problems and goals in the sections above.

Finally, even if you don't have any coding experience, you are still able, and very welcome, to help out! Since all rated Crazyhouse games are put into the Lichess Database, the best way is to simply play more games online. Hopefully, there will be a bot account to measure its strengths and weaknesses.

CONCLUSION
Thank you so much for reading this far! I hope you've enjoyed the explanation, and look forward to working together and creating a community-developed learning Crazyhouse engine a reality.
