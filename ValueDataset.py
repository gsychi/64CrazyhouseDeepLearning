import torch.utils.data as data_utils
import torch.nn as nn
import torch
import numpy as np
import h5py

# change to h5py storage

class ValueDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, outputs):
        self.features = inputs
        self.targets = outputs


    def __getitem__(self, index):
        return self.features[index], np.expand_dims(self.targets[index],axis=0)


    def __len__(self):
        return len(self.features)


