import numpy as np
import h5py

bigInputs = np.zeros((5845235, 1, 32, 28))

with h5py.File('Training Data/17-01masterInputs.h5', 'r') as hf:
            inputs = hf["Inputs"][:]
print(inputs.shape)

for i in range(len(inputs)):
    bigInputs[i] = inputs[i]
    print(i)
inputs = []

with h5py.File('Training Data/17-02masterInputs.h5', 'r') as hf:
            inputs = hf["Inputs"][:]
print(inputs.shape)

for i in range(len(inputs)):
    bigInputs[1410605+i] = inputs[i]
inputs = []

with h5py.File('Training Data/17-03masterInputs.h5', 'r') as hf:
            inputs = hf["Inputs"][:]
print(inputs.shape)

for i in range(len(inputs)):
    bigInputs[2979606+i] = inputs[i]
inputs = []

with h5py.File('Training Data/17-04masterInputs.h5', 'r') as hf:
            inputs = hf["Inputs"][:]
print(inputs.shape)

for i in range(len(inputs)):
    bigInputs[4351041+i] = inputs[i]
inputs = []

"""
for j in range(0, 24):
    with h5py.File('Training Data/Full Data/bigInputs(17-06)-(18-10).h5', 'r') as hf:
                inputs2 = hf["Inputs"][j*250000:(j+1)*250000]
    for i in range(len(inputs2)):
        bigInputs[1606525+j*250000+i] = inputs2[i]
        #print(1606525+j*250000+i)
    inputs2 = []

with h5py.File('Training Data/Full Data/bigInputs(17-06)-(18-10).h5', 'r') as hf:
    inputs2 = hf["Inputs"][6000000:]
for i in range(len(inputs2)):
    bigInputs[7606525+i] = inputs2[i]
    print(7606525+i)
inputs2 = []
"""

with h5py.File('Training Data/Full Data/bigInputs(17-01)-(17-04).h5', 'w') as hf:
    hf.create_dataset("Inputs", data=bigInputs, compression='gzip', compression_opts=9)
