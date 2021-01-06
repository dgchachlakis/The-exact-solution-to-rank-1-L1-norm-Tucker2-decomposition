import numpy as np
import itertools
import math
import utils
def computeCandidates(matrix,half_sphere=True):
    rho=matrix.shape[0]
    N=matrix.shape[1]
    assert rho<=N,'Input matrix should have full-row rank.'
    numOfambiguities=rho-1
    Bpool=utils.decimal2binary(list(range(2**numOfambiguities)),numOfambiguities)
    combinations=list(itertools.combinations((range(N)), numOfambiguities))
    candidates=set()
    for combination in combinations:
        matrixI=matrix[:,combination]
        factorU,diagS,factorVt=np.linalg.svd(matrixI, full_matrices=True)
        v=factorU[:,-1]
        b_ambiguous=utils.mysign(matrix.T@v).flatten()
        #print(b_ambiguous)
        for n in range(Bpool.shape[1]):
            b=b_ambiguous.copy()
            try:
                b[b==0]=Bpool[:,n].copy()
            except:
                print('Check tolerance of mysign function.')
                exit()
            b=tuple(b[0]*b)
            candidates.add(b)
    if not half_sphere:
        L=candidates.shape[1]
        for n in range(L):
            b=np.array(candidates[n])
            candidates.add(tuple(-b))
    return np.array(list(candidates)).T