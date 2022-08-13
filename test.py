import torch
import numpy as np

data = np.ones((2, 2))
temp = data[:, 0]
temp *= 2
print(temp)
print(data)