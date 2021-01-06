import numpy as np
def l1tucker2metric(tensor,factorU,factorV):
    assert np.ndim(tensor)==3,'Expected a 3-way array as input.'
    D=tensor.shape[0]
    M=tensor.shape[1]
    N=tensor.shape[2]
    assert factorU.shape[1]<=factorU.shape[0]<=D,'Check size of factor U'
    assert factorV.shape[1]<=factorV.shape[0]<=M,'Check size of factor V'
    met=0
    for n in range(N):
        Z=factorU.T@tensor[:,:,n]@factorV
        met+=np.sum(np.abs(Z.flatten()))
    return met

    