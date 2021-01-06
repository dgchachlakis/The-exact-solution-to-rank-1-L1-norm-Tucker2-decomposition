import numpy as np
def xmatrix(tensor):
    assert np.ndim(tensor)==3,'Expected a 3-way array as input.'
    D=tensor.shape[0]
    M=tensor.shape[1]
    N=tensor.shape[2]
    X=np.zeros((D,M*N))
    for n in range(N):
        X[:,n*M:(n+1)*M]=tensor[:,:,n]
    return X