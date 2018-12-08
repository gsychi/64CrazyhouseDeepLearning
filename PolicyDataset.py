import torch.utils.data as data_utils
import torch.nn as nn
import torch
import numpy as np
import h5py


# change to h5py storage

class PolicyDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, outputs):
        self.features = inputs
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


