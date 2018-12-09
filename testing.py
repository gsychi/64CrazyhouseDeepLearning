import h5py
import numpy as np

#with h5py.File('Training Data/17-03masterOutputs.h5', 'r') as hf:
            #outputs = hf["Outputs"][1371435:]

outputs = np.load("Training Data/Full Data/bigOutputsArgmax(17-03)-(18-10).npy")[1371435:]
outputs2 = np.load("Training Data/Full Data/bigOutputsArgmax(17-04)-(18-10).npy")

for i in range(len(outputs)):
    print(outputs[i]==outputs2[i])
