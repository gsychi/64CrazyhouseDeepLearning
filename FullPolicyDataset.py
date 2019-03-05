import torch.utils.data as data_utils
import torch.nn as nn
import torch
import numpy as np
import h5py

# change to h5py storage

class FullPolicyDataset(torch.utils.data.Dataset):

    def __init__(self):
        print("Full Policy Dataset!")


    def __getitem__(self, index):
        """
        Collecting Data from 18-01 to 18-11.
        """
        if index < 1625167:
             with h5py.File('Training Data/17-12PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][index]
             with h5py.File('Training Data/17-12Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 3093406:
             with h5py.File('Training Data/18-01PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-1625167+index]
                
             with h5py.File('Training Data/18-01Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-1625167+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 4757973:
             with h5py.File('Training Data/18-02PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-3093406+index]
                
             with h5py.File('Training Data/18-02Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-3093406+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 6569877:
             with h5py.File('Training Data/18-03PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-4757973+index]
                
             with h5py.File('Training Data/18-03Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-4757973+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 8418049:
             with h5py.File('Training Data/18-04PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-6569877+index]
                
             with h5py.File('Training Data/18-04Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-6569877+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 10370599:
             with h5py.File('Training Data/18-05PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-8418049+index]
                
             with h5py.File('Training Data/18-05Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-8418049+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 12000088:
             with h5py.File('Training Data/18-06PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-10370599+index]
                
             with h5py.File('Training Data/18-06Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-10370599+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 13555217:
             with h5py.File('Training Data/18-07PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-12000088+index]
                
             with h5py.File('Training Data/18-07Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-12000088+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 15140394:
             with h5py.File('Training Data/18-08PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-13555217+index]
                
             with h5py.File('Training Data/18-08Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-13555217+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 16572740:
             with h5py.File('Training Data/18-09PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-15140394+index]
                
             with h5py.File('Training Data/18-09Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-15140394+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        elif index < 17985687:
             with h5py.File('Training Data/18-10PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-16572740+index]
                
             with h5py.File('Training Data/18-10Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-16572740+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)
        else:
             with h5py.File('Training Data/18-11PolicyOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-17985687+index]
                
             with h5py.File('Training Data/18-11Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-17985687+index]
             return inputs, torch.from_numpy(actions).type(torch.LongTensor)

    def __len__(self):
        return 19542203
