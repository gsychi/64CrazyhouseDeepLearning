import numpy as np
import h5py

bigOutputs = np.zeros(5845235)

with h5py.File('Training Data/17-01masterOutputs.h5', 'r') as hf:
            outputs = hf["Outputs"][:]
with h5py.File('Training Data/17-02masterOutputs.h5', 'r') as hf:
            outputs2 = hf["Outputs"][:]
with h5py.File('Training Data/17-03masterOutputs.h5', 'r') as hf:
            outputs3 = hf["Outputs"][:]
with h5py.File('Training Data/17-04masterOutputs.h5', 'r') as hf:
            outputs4 = hf["Outputs"][:]

for i in range(len(outputs)):
    bigOutputs[i] = outputs[i]
print(len(outputs))
for i in range(len(outputs2)):
    bigOutputs[1410605+i] = outputs2[i]
print(len(outputs2))
for i in range(len(outputs3)):
    bigOutputs[2979606+i] = outputs3[i]
print(len(outputs3))
for i in range(len(outputs4)):
    bigOutputs[4351041+i] = outputs4[i]
print(len(outputs4))

np.save("Training Data/Full Data/bigOutputsArgmax(17-01)-(17-04).npy", bigOutputs)
