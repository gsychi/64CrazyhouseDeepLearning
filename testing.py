import numpy as np

inList = [[3,2],[2,1],[5,6],[7,8],[2,9]]
outList = [4,5,6,1,9]

inputs = np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])
outputs = np.array([0,0,0,0,0])

i = 0
while len(inList) > 0:
    inputs[i] = inList[len(inList)-1]
    outputs[i] = outList[len(inList)-1]
    inList.pop()
    outList.pop()
    i += 1

print(inputs)
print(outputs)
