import numpy as np

def sparsity(m):
    return 1 - np.count_nonzero(m) / m.size
