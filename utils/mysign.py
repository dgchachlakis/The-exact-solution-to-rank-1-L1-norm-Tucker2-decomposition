import numpy as np
def mysign(x, tol = 1e-7):
    x[np.abs(x) < tol] = 0
    return np.sign(x)