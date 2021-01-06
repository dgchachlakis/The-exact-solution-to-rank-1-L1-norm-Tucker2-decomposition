import numpy as np
import tensorly as tl
from utils import *
def exact(tensor):
    assert np.ndim(tensor)==3,'Expected a 3-way array as input.'
    X=xmatrix(tensor)
    D=tensor.shape[0]
    M=tensor.shape[1]
    N=tensor.shape[2]
    B=decimal2binary(list(range(2**N)),N)
    metopt=0
    for n in range(B.shape[1]):
        b=B[:,n,None]
        Z=X@np.kron(b,np.eye(M))
        sigmamax=np.linalg.svd(Z)[1].flatten()[0]
        if sigmamax>metopt:
            metopt=sigmamax
            bopt=b
    U,S,Vt=np.linalg.svd(X@np.kron(bopt,np.eye(M)))
    uopt=U[:,0]
    vopt=Vt[0,:]
    return uopt,vopt,bopt,metopt