import numpy as np
def decimal2binary(decimal, numberOfBits):
   N = len(decimal)
   B = np.zeros((numberOfBits, N))
   for n in range(N):
      B[:, n] = np.array(list(np.binary_repr(n).zfill(numberOfBits))).astype(np.int8)
   return 2*B-1