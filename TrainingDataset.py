import torch.utils.data as data_utils
import torch.nn as nn
import torch
import numpy as np
import h5py


# change to h5py storage

class TrainingDataset(torch.utils.data.Dataset):

    def __init__(self, directory, outputs):
        with h5py.File(directory, 'r') as hf:
            self.features = hf["Inputs"][0:1000000]
            print(len(self.features))

        self.targets = outputs
        self.numpy = outputs.numpy()


    def __getitem__(self, index):
        # output vector created
        array = np.zeros(4504)
        array[int(self.numpy[index])] = 1
        output = torch.from_numpy(array)
        return self.features[index], output


    def __len__(self):
        return len(self.features)


