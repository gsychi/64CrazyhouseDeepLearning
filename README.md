# 64 Crazyhouse Deep Learning
A deep learning Crazyhouse chess program that uses a Monte Carlo Tree Search (MCTS) based evaluation system and reinforcement to enhance its play style. Created as an individual project by a high school student.

Information on the inner workings and the code breakdown can be found on the blog post [here](https://ftlearning.wordpress.com/2018/08/13/64-a-crazyhouse-learning-project/). 

Side Note: Parts of the code are yet to be optimized since project is still in progress.


## What is this?

This is a framework of a neural network-based Crazyhouse Chess Engine inspired by the procedures specified by Google DeepMind and their multiple papers on AlphaGo and AlpahZero. This project was initiated in June 15, 2018, following the successful framework of a self-learning Tic Tac Toe engine that managed to solve the game in an hour. You may find the repository for that [here](https://github.com/FTdiscovery/GOMCTS); however, that project has yet to be annotated for the general public to use. 

Currently, this repository is designed for three processes: supervised learning (training a network model based on PGN master games), reinforcement learning (having the best network play itself to generate training games, from which this data can be used to train better networks), as well as self-learning (start with a randomly initialized or even a pre-trained Pytorch model if desired, from which training games and new networks will be created for however long the user desires.)

As of now, due to the lack of computational resources readily available for me, the focus is on creating a trained model based on top available games. From this documentation, I will explain the whole process of training a model and sending it back for further testing. 

Your name will be added on the list of contributors once you have uploaded a relevant file. :)

## Requirements

It will be necessary to download the following packages:

python-chess <br>
scipy <br>
numpy (if that's somehow not on your computer) <br>
torch (pytorch) <br>
torchvision <br>
pathlib <br>

As this program was written, tested, and edited on the PyCharm CE IDE, I also recommend that users also download the interpreter and run the code on it. However, I acknowledge that it is possible to run any of the attached files through terminal/other IDEs, provided that all the necessary libraries are downloaded. Python 3 (3.6) is strongly recommended.


Once these are all running on your computer, then you will be able to continue to the next step!

## How do I run the program?

Start off by downloading the repository through terminal...

    git clone https://github.com/FTdiscovery/64CrazyhouseDeepLearning.git
    
... or by downloading the ZIP file on the top right hand corner.

## Creating a Supervised Learning (SL) Based Model


### Creating a Database

As you may imagine, this step will require a database created from Crazyhouse games. There exists no database or folder for that attached in this repository; luckily, this process can be easily completed by following the steps on the documentation. The <b> CreateDatabase.py </b> file looks through all the PGN files in a folder and generates two numpy files for the neural network to train from. For that reason, it is important to first create two (empty) folders in the 64CrazyhouseDeepLearning repository. One will be named <i>lichessdatabase</i>, while the other will be named <i>Training Data</i>. You may change the names of them, of course - these are just the directories I have specified in the program - but bear in mind that it is necessary to change the code if you desire so.

Once these folders are created, we begin by downloading Crazyhouse pgn files into the <i> lichessdatabase </i> folder. Zipped PGN files containing ~130-190,000 Crazyhouse games can be downloaded [from the Lichess Database.](https://database.lichess.org/). These are found by clicking on the VARIANTS tab, and then scrolling down to the CRAZYHOUSE section. You may download as many files and drag them into the <i> lichessdatabase </i> folder. Do note that pgn files from the above database have to be unzipped.

Once all files are successfully placed in the database folder, simply run <b> CreateDatabase.py</b>. The file will take some time to generate a database. Be careful, however, of how much memory and RAM you have on your computer. A dataset of ~15,000 games creates an input matrix and output matrix with ~830,000 rows, and this requires 38.37 GB of space. Too large of a dataset may result in a SIGKILL signal. 

### Determining the Network Architecture

The second step is to determine the model parameters that you would like to use. This can be found and edited on <b> ChessConvNet.py</b>. The default setting at the moment is a 5 layer convolutional neural network, with each layer having 32 convolutions. You are free to use the current architecture or edit it however you like (i.e. add more layers, have more convolutions per layer, change kernel size...). I suggest to not increase the number of layers, but rather, to increase the width of the network. 
    
```python
class ChessConvNet(nn.Module):
    def __init__(self, num_classes):
        self.numClasses = num_classes
        super(ChessConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # 1, 32
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),  # 32, 32
            nn.BatchNorm2d(32),  # 32
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32, 64
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64, 64
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64, 64
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc = nn.Linear(896 * 64, num_classes)
```

Here is an example of a modification of the neural network. The first two layers have 32 convolutions per layer, whereas the last three layers (excluding the fully connected layer) have 64 convolutions per layer. For those less familiar with PyTorch, it may be useful to look at self.layer3 to understand how two layers are connected.

### Training the Network

Once all of this is completed, you can simply run <b>TrainNetwork.py</b> for as long as possible. The Pytorch model will be periodically saved on the repository folder you have downloaded. You should upload them onto the[Google Drive folder](https://drive.google.com/drive/folders/1o8uzyvG1zVRAqnvbdFzHs6MHN2eUm0PP) once your network has been trained for over 10-20 epochs.

If you are having troubles locating the downloaded file, you can change the name of the saved model by changing the saveDirectory="" on the last line of the file. Similarly, you may load a pretrained model in the loadDirectory="" on the same line. As an example, the default setting below loads a file from nothing.pt (the code does nothing if this directory does not exist), and saves the newly trained model as sFULL-5LAYER-32.pt.

```python
savedtrainNetwork(inputs, outputs, loadDirectory="nothing.pt", saveDirectory="sFULL-5LAYER-32.pt", EPOCHS=1000,
                 BATCH_SIZE=64, updateInterval=99, LR=0.01)
 ```

<br>

### When uploading into the Google Drive Folder, it is important to follow a naming protocol in order to track each network. 

<b> If your network has the same number of convolutions on each layer </b>, please name your file as the following:

+ sUSERNAME-[number of layers]-[number of convolutions per layer].pt

i.e. If I use the default settings, my file would be uploaded as <b>sFTDiscovery-5LAYER-32.pt</b>. 

<b> If your network DOES NOT have the same number of convolutions on each layer </b>, please name your file as the following:

+ sUSERNAME-[number of layers]-[number of convolutions per layer].pt

i.e. If I use the above modification, my file would be uploaded as <b>sFTDiscovery-5LAYER-32-32-64-64-64.pt</b>.

<b> If you decide to completely overhaul the architecture</b>, please add a file of the changed ChessConvNet.py and name your files as the following:

+ sUSERNAME.pt (for the trained model)
+ sUSERNAME.py (for the changed ChessConvNet.py architecture)

i.e. If I created a completely new network architecture on ChessConvNet.py. I would then call that <b> FTDiscovery.py</b> and send my model, <b> sFTDiscovery.pt </b>, along with it.


### What about ChessResNet.py?

ChessResNet.py has yet to be tested. For those who would like to try it out, you will have to edit the TrainNetwork function and change the line...

```python
model = ChessConvNet(OUTPUT_ARRAY_LEN).double()
```

...into this.
```python
model = ChessResNet.ResNet18().double()

# Note that ChessResNet.ResNet32().double(), ChessResNet.ResNet50().double(), ChessResNet.ResNet101().double(), and ChessResNet.ResNet152().double() are possible architectures.
```

Saved models would be specified as followed:

sUSERNAME-RESNET[resnet architecture number].pt

i.e. A ResNet32 model trained by username LinguisticBobby would be called sLinguisticBobby-RESNET32.pt


## What if I have a GPU?

A GPU is definitely useful in speeding up the training of the neural network. Unfortunately, since I have yet to test the code with one, I am unable to verify whether <b>GPUTrainNetwork.py</b> works. Otherwise, you should follow the exact same process as earlier, with the exception of running the program on GPUTrainNetwork instead of TrainNetwork.py.

## Reinforcement Learning Based Model

Instructions coming soon...



## Required Future Edits

1. GPU Optimization (Urgency: <b>High</b>)
2. Changes to playout mechanism. Right now, all playouts end only after a result (win or draw) is reached. However, in the future, the tree should be able to specify a search depth and use np.amax to determine the win probability in the final searched position (1-np.amax if depth is even). This has not been implemented yet, as all values outputted by the neural network are less than 1e-03 after the first ten moves or so. (Urgency: <b>High</b>)
3. Create a Leela-like online server training system for everyone to work with and to lend their GPUs on. (Urgency: <b>medium</b>)
4. Time Control – If this engine is to compete in competitions, it must be able to determine when it can search further, and whether it has the time to do so.(Urgency: <b>low</b>)
5. Translate into C++? (Urgency: <b>low</b>)

## Additional Code Information

### Input Representation
The input for the neural network is a basic (raw) representation of the board. It is an array of 896 values (14 planes of 8×8 boards), which determines the player turn, position of each piece type, captive pieces held by a player, and whether the piece is a promotion. The last part is relevant since promoted pieces can only be placed down as pawns after being captured.

### Output Representation
The output for the neural network is a 1D representation of multiple move planes (8×8 boards). Here they are as followed:

The first five planes represent the possible squares in which a given piece should be placed (pawn, knight, bishop, rook, queen).

The next plane represents the piece that should be picked up from the board.

The next 56 planes represent the possible queen moves. There are 8 planes for each direction (N, NE, E, SE, S, SW, W, NW), and 7 planes for how far a piece can move (1-7). This is used to map all moves made by a pawn, bishop, rook, queen, and king.

The next 8 planes represent the possible knight moves (2F1L, 2F1R, 1F2L, 1F2R, 1B2L, 1B2R, 2B1L, 2B1R). This is used to map moves made by a knight.

Last, we have 3 planes of 8×1 boards, which determine the column in which a pawn is being underpromoted. Each plane denotes a different promotion (rook, bishop, knight).

In total, the action is represented by an array of (5+1+56+8) x (8×8) + 3×8 = 4504 values.

## Sample Games


### Self Play Games


1. e4 e6 2. d4 d5 3. e5 Ne7 4. Nf3 Nf5 5. Bd3 Nc6 6. O-O Be7 7. c3 O-O 8. Bf4 f6 9. Nbd2 fxe5 10. dxe5 P@e4 11. P@d6 exf3 12. Nxf3 Bxd6 13. exd6 P@e4 14. Bxe4 dxe4 15. P@g5 exf3 16. gxf3 N@h4 17. B@h3 N@e2+ 18. Kh1 B@g2+ 19. Bxg2 Nxg2 20. Kxg2 Nh4+ 21. Kh1 B@g2# 0-1


1. e4 Nc6 2. Nc3 Nf6 3. Nf3 d5 4. exd5 Nxd5 5. Bc4 e6 6. O-O Nb6 7. Bb5 Bd7 8. Re1 Be7 9. d4 O-O 10. P@h6 gxh6 11. Bxh6 P@e4 12. P@g7 exf3 13. Qxf3 P@g4 14. Qxg4 N@f5 15. gxf8=Q# 1-0


1. e4 Nc6 2. Nc3 Nf6 3. Nf3 d5 4. exd5 Nxd5 5. d4 e6 6. Bb5 Bb4 7. Bd2 Bxc3 8. bxc3 N@e4 9. @b2 Nxd2 10. Qxd2 B@h5 11. O-O-O O-O 12. Kb1 @g4 13. Ng5 Bg6 14. Nxf7 Rxf7 15. Bxc6 bxc6 16. N@h6+ gxh6 17. Qxh6 Rf6 18. N@e7+ Kh8 19. B@g7# 1-0

### Wins against Humans

1. e4 Nf6 2. e5 d5 3. d4 Ne4 4. f3 Bf5 5. fxe4 dxe4 6. Bc4 e6 7. Nc3 Qh4+ { White resigns. } 0-1

1. e4 e5 2. Nf3 Nc6 3. Bc4 d6 4. Nc3 Be7 5. d4 exd4 6. Nxd4 Nxd4 7. Qxd4 P@h3 8. Bxf7+ Kxf7 9. Qd5+ B@e6 10. Qh5+ g6 11. Qf3+ Nf6 12. N@g5+ Kg7 13. P@h6+ Kxh6 14. Nxe6+ g5 15. Bxg5+ Kg6 16. Qf5+ Kf7 17. Nxd8+ Bxd8 18. B@d5+ P@e6 19. Bxe6+ Bxe6 20. Qxe6+ Kxe6 21. P@f5+ Kf7 22. P@e6+ { Black resigns. } 1-0





## Contributors

Below is a list of contributors to the project:
