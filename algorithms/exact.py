"""
Exact solution to rank-1 L1-norm Tucker2 decomposition by exhaustive search.
"""
__author__='D. G. Chachlakis'
import numpy as np
import tensorly as tl
from utils import *
def exact(tensor):
    """ 
    Input: 
    ------
    3-way tensor of size D-by-M-by-N--i.e., a collection of 2-way measurements across mode 3

    Output: 
    ------
    uopt: left-hand side optimal basis vector
    vopt: right-hand side optimal basis vector
    bopt: an optimal antipodal binary vector
    metopt: the maximum attainable metric
    numOfCandidates: number of antipodal binary vectors examined
    """
    assert np.ndim(tensor)==3,'Expected a 3-way array as input.'
    X=xmatrix(tensor)
    D=tensor.shape[0]
    M=tensor.shape[1]
    N=tensor.shape[2]
    B=decimal2binary(list(range(2**N)),N)
    numOfCandidates=B.shape[1]
    metopt=0
    for n in range(numOfCandidates):
        b=B[:,n,None]
        Z=X@np.kron(b,np.eye(M))
        sigmamax=np.linalg.svd(Z)[1].flatten()[0]
        if sigmamax>metopt:
            metopt=sigmamax
            bopt=b
    U,S,Vt=np.linalg.svd(X@np.kron(bopt,np.eye(M)))
    uopt=U[:,0]
    vopt=Vt[0,:]
    return uopt,vopt,bopt,metopt,numOfCandidates