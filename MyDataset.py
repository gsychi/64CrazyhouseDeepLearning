import torch.utils.data as data_utils
import torch.nn as nn
import torch

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, outputs):
        self.features = inputs
        self.targets = outputs

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)
