# 64CrazyhouseDeepLearning
A deep learning Crazyhouse chess program that uses a Monte Carlo Tree Search (MCTS) based evaluation system and reinforcement to enhance its play style.


## What is this?

This is a framework of a neural network-based Crazyhouse Chess Engine inspired by the procedures specified by Google DeepMind and their multiple papers on AlphaGo and AlpahZero. This project was initiated in June 15, 2018, following the successful framework of a self-learning Tic Tac Toe engine that managed to solve the game in an hour. You may find the repository for that [here](https://github.com/FTdiscovery/GOMCTS); however, that project has yet to be annotated for the general public to use. 

More information on the inner workings and the code breakdown can be found on the blog post [here](https://ftlearning.wordpress.com/2018/08/13/64-a-crazyhouse-learning-project/). 

Currently, this repository is designed for three processes: supervised learning (training a network model based on PGN master games), reinforcement learning (having the best network play itself to generate training games, from which this data can be used to train better networks), as well as self-learning (start with a randomly initialized or even a pre-trained Pytorch model if desired, from which training games and new networks will be created for however long the user desires.)

As of now, due to the lack of computational resources readily available for me, the focus is on creating a trained model based on top available games. From this documentation, I will explain the whole process of training a model and sending it back for further testing. 

Your name will be added on the list of contributors once you have uploaded a relevant file. :)

### Requirements

### How to Run

### What if I have a GPU?

### Required Future Edits

1. GPU Optimization (Urgency: High)
2. Changes to playout mechanism. Right now, all playouts end only after a result (win or draw) is reached. However, in the future, the tree should be able to specify a search depth and use np.amax to determine the win probability in the final searched position (1-np.amax if depth is even). This has not been implemented yet, as all values outputted by the neural network are less than 1e-03 after the first ten moves or so. (Urgency: High)
Create a Leela-like online server training system for everyone to work with and to lend their GPUs on. (Urgency: medium)
Time Control – If this engine is to compete in competitions, it must be able to determine when it can search further, and whether it has the time to do so.(Urgency: low)
Translate into C++? (Urgency: low)

## Contributors

Below is a list of contributors to the project:
