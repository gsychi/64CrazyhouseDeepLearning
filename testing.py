import numpy as np
from scipy.interpolate import interp1d

outputs = np.array([1, 0.5, 0])
outputs = (outputs / 1.0002) + 0.0001
outputs = np.log((outputs / (1 - outputs)))
print(outputs)

haha = ["yes", "nah"]
ah = str(haha)
print(ah)
f = open("testing.txt", "w+")
f.write(ah)
f.close()

