import numpy as np
def ymatrix(tensor):
    assert np.ndim(tensor) == 3, 'Expected a 3-way array as input.'
    D = tensor.shape[0]
    M = tensor.shape[1]
    N = tensor.shape[2]
    Y = np.zeros((D*M, N))
    for n in range(N):
        Y[:, n] = tensor[:, :, n].flatten()
    return Y