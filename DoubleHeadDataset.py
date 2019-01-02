import torch.utils.data as data_utils
import torch.nn as nn
import torch
import numpy as np
import h5py

class DoubleHeadDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, policyOut, valueOut):
        self.features = inputs
        self.targets = policyOut   # .type(torch.LongTensor) for nll loss
        self.targets2 = valueOut
        self.numpy = policyOut.numpy()

    def __getitem__(self, index):

        # output vector created
        array = np.zeros(4504)
        array[int(self.numpy[index])] = 1
        output = torch.from_numpy(array)
        return self.features[index], output, np.expand_dims(self.targets2[index], axis=0)

    def __len__(self):
        return len(self.features)


