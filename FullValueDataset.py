import torch.utils.data as data_utils
import torch.nn as nn
import torch
import numpy as np
import h5py

# change to h5py storage

class FullValueDataset(torch.utils.data.Dataset):

    def __init__(self):
        print("Full Value Dataset!")


    def __getitem__(self, index):
        """
        Collecting Data from 18-01 to 18-11.
        """
        if index < 1625167:
             with h5py.File('Training Data/17-12ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][index]
             with h5py.File('Training Data/17-12Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 3093406:
             with h5py.File('Training Data/18-01ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-1625167+index]
                
             with h5py.File('Training Data/18-01Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-1625167+index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 4757973:
             with h5py.File('Training Data/18-02ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-3093406+index]
                
             with h5py.File('Training Data/18-02Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-3093406+index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 6569877:
             with h5py.File('Training Data/18-03ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-4757973+index]
                
             with h5py.File('Training Data/18-03Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-4757973+index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 8418049:
             with h5py.File('Training Data/18-04ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-6569877+index]
                
             with h5py.File('Training Data/18-04Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-6569877+index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 10370599:
             with h5py.File('Training Data/18-05ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-8418049+index]
                
             with h5py.File('Training Data/18-05Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-8418049+index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 12000088:
             with h5py.File('Training Data/18-06ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-10370599+index]
                
             with h5py.File('Training Data/18-06Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-10370599+index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 13555217:
             with h5py.File('Training Data/18-07ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-12000088+index]
                
             with h5py.File('Training Data/18-07Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-12000088+index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 15140394:
             with h5py.File('Training Data/18-08ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-13555217+index]
                
             with h5py.File('Training Data/18-08Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-13555217+index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 16572740:
             with h5py.File('Training Data/18-09ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-15140394+index]
                
             with h5py.File('Training Data/18-09Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-15140394+index]
             return inputs, np.expand_dims(actions, axis=0)
        elif index < 17985687:
             with h5py.File('Training Data/18-10ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-16572740+index]
                
             with h5py.File('Training Data/18-10Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-16572740+index]
             return inputs, np.expand_dims(actions, axis=0)
        else:
             with h5py.File('Training Data/18-11ValueOutputs.h5', 'r') as hf:
                actions = hf["Outputs"][-17985687+index]
                
             with h5py.File('Training Data/18-11Inputs.h5', 'r') as hf:
                inputs = hf["Inputs"][-17985687+index]
             return inputs, np.expand_dims(actions, axis=0)

    def __len__(self):
        return 19542203
