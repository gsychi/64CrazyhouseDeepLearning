#!/usr/bin/python

import h5py

with h5py.File("Training Data/INTEGER_StockfishDataset_INPUTS.h5", 'r') as hf:
        boards = hf["Inputs"][:]
        print(len(boards))

print(boards[0])
print(boards[1])

"""
board = ChessEnvironment()
network = MCTS("New Networks/smallnet.pt", 3)

class myThread(threading.Thread):
    def __init__(self, network, board):
        threading.Thread.__init__(self)
        self.board = board
        self.network = copy.deepcopy(network)
    def run(self):
        self.network.competitivePlayoutsFromPosition(1, board)

start = time.time()
threads = []
for i in range(10):
    t = myThread(network, board)
    threads.append(t)
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
end = time.time()
print(end-start)

start = time.time()
network.competitivePlayoutsFromPosition(10, board)
end = time.time()
print(end-start)

"""


