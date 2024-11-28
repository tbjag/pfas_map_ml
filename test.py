import numpy as np

# Sample NumPy array
data = np.array([10, 20, 30, 40, 50])

# Min-max normalization
data_normalized = (data - data.min()) / (data.max() - data.min())

print(data_normalized)
