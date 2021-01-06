import numpy as np
def procrustes(matrix,numOfComponents):
    assert matrix.shape[1]==np.linalg.matrix_rank(matrix)==numOfComponents, 'Procrustes is expecting a matrix with full-column rank as input.'
    U,S,Vt=np.linalg.svd(matrix)
    return U@Vt