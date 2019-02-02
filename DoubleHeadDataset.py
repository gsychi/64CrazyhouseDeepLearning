import torch.utils.data as data_utils
import torch.nn as nn
import torch
import numpy as np
import h5py
import ActionToArray

class DoubleHeadTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, policyOut, valueOut):
        self.features = inputs
        self.targets = policyOut   # .type(torch.LongTensor) for nll loss
        self.targets2 = valueOut
        self.numpy = policyOut.numpy()

    def __getitem__(self, index):

        # CREATE INPUT VECTOR

        #960String Method!!
        #inArray = np.zeros(960)
        #for i in range(960):
        #    inArray[i] = int(self.features[index][i])
        #inArray = np.reshape(inArray, (15, 8, 8))

        #BinaryConverted Method!!
        inArray = ActionToArray.binaryArrayToBoard(self.features[index])

        # policy output vector created
        array = np.zeros(2768)
        array[int(self.numpy[index])] = 1
        output = torch.from_numpy(array)

        return inArray, output, np.expand_dims(self.targets2[index], axis=0)

    def __len__(self):
        return len(self.features)

class DoubleHeadDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, policyOut, valueOut):
        self.features = inputs
        self.targets = policyOut   # .type(torch.LongTensor) for nll loss
        self.targets2 = valueOut
        self.numpy = policyOut.numpy()

    def __getitem__(self, index):

        # output vector created
        array = np.zeros(2768)
        array[int(self.numpy[index])] = 1
        output = torch.from_numpy(array)
        return self.features[index], output, np.expand_dims(self.targets2[index], axis=0)

    def __len__(self):
        return len(self.features)


