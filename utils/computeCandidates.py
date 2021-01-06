import numpy as np
import itertools
import math
import utils
def computeCandidates(matrix,halfSphere=True):
    rho=matrix.shape[0]
    N=matrix.shape[1]
    assert rho<=N,'Input matrix should have full-row rank.'
    numOfambiguities=rho-1
    Bpool=utils.decimal2binary(list(range(2**numOfambiguities)),numOfambiguities)
    combinations=list(itertools.combinations((range(N)),numOfambiguities))
    candidates=set()
    for combination in combinations:
        matrixI=matrix[:,combination]
        factorU,diagS,factorVt=np.linalg.svd(matrixI, full_matrices=True)
        v=factorU[:,-1]
        b_ambiguous=utils.mysign(matrix.T@v).flatten()
        for n in range(Bpool.shape[1]):
            b=b_ambiguous.copy()
            try:
                b[b==0]=Bpool[:,n].copy()
            except:
                print('Check tolerance of mysign function.')
                exit()
            b=tuple(b[0]*b)
            candidates.add(b)
    if not halfSphere:
        otherHalf=set()
        for b in candidates:
            otherHalf.add(tuple(-np.array(b)))
        candidates=candidates.union(otherHalf)
    return np.array(list(candidates)).T