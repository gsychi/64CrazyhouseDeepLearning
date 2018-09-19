# 64CrazyhouseDeepLearning
A deep learning Crazyhouse chess program that uses a Monte Carlo Tree Search (MCTS) based evaluation system and reinforcement to enhance its play style.

2111 lines

https://ftlearning.wordpress.com/2018/08/13/64-a-crazyhouse-learning-project/

## What is this?

This is a framework of a neural network-based Crazyhouse Chess Engine inspired by the procedures specified by Google DeepMind and their multiple papers on AlphaGo and AlpahZero. This project was initiated in June 15, 2018, following the successful framework of a self-learning Tic Tac Toe engine that managed to solve the game in an hour. You may find the repository for that [here](https://github.com/FTdiscovery/GOMCTS); however, that project has yet to be annotated for the general public to use. 

Currently, this repository is designed for three processes: supervised learning (training a network model based on PGN master games), reinforcement learning (having the best network play itself to generate training games, from which this data can be used to train better networks), as well as self-learning (start with a randomly initialized or even a pre-trained Pytorch model if desired, from which training games and new networks will be created for however long the user desires.)

As of now, due to the lack of computational resources readily available for me, the focus is on creating a trained model based on top available games. From this documentation, I will explain the whole process of training a model and sending it back for further testing. 

Your name will be added on the list of contributors once you have uploaded a relevant file. :)

### Requirements

### How to Run

### What if I have a GPU?

### Code

## Contributors

Below is a list of contributors to the project:
