import numpy as np
import tensorly as tl
from algorithms import *
import time

print(50*'\n')
tensor=np.random.randn(4,4,10)

start_time = time.time()
uopt_exact,vopt_exact,bopt_exact,metopt_exact=exact(tensor)
print("--- %s seconds ---" % (time.time() - start_time))
print(metopt_exact)
print(uopt_exact)

start_time = time.time()
uopt_exactpoly,vopt_exactpoly,bopt_exactpoly,metopt_exactpoly=exactpoly(tensor)
print("--- %s seconds ---" % (time.time() - start_time))

print(metopt_exactpoly)
print(uopt_exactpoly)