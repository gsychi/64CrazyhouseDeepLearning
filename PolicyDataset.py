import torch.utils.data as data_utils
import torch.nn as nn
import torch
import numpy as np
import h5py


# change to h5py storage

class PolicyDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, outputs):
        self.features = inputs
        self.targets = outputs   # .type(torch.LongTensor) for nll loss
        self.numpy = outputs.numpy()


    def __getitem__(self, index):

        # THIS IS USED WHEN POISSONNLLLOSS
        # output vector created
        array = np.zeros(4504)
        array[int(self.numpy[index])] = 1
        output = torch.from_numpy(array)
        return self.features[index], output


        """
        # THIS IS USED WHEN NLLLOSS
        return self.features[index], self.targets[index]
        """

    def __len__(self):
        return len(self.features)


