import numpy as np
import algorithms as rank1l1tucker2
import time
# Example 1
tensor=np.random.randn(2,2,16)
print("=======Example 1===========\n")
start_time=time.time()
uopt_exact,vopt_exact,bopt_exact,metopt_exact,numOfCandidates0=rank1l1tucker2.exact(tensor)
execution_time=time.time() - start_time
print('\t1. Solution by exhaustive search')
print('\tMetric: \t\t\t', metopt_exact)
print('\tExecution time (s): \t\t', execution_time)
print('\tNumber of candidates examined: \t',numOfCandidates0,'\n')
start_time=time.time()
uopt_exactpoly,vopt_exactpoly,bopt_exactpoly,metopt_exactpoly,numOfCandidates1=rank1l1tucker2.exactpoly(tensor,halfSphere=True)
execution_time=time.time() - start_time
print('\t2. Solution by intelligent search over halh sphere')
print('\tMetric: \t\t\t', metopt_exactpoly)
print('\tExecution time (s): \t\t', execution_time)
print('\tNumber of candidates examined: \t',numOfCandidates1,'\n')
start_time=time.time()
uopt_exactpoly2,vopt_exactpoly2,bopt_exactpoly2,metopt_exactpoly2,numOfCandidates2=rank1l1tucker2.exactpoly(tensor,halfSphere=False)
execution_time=time.time() - start_time
print('\t3. Solution by intelligent search over full sphere')
print('\tMetric: \t\t\t', metopt_exactpoly2)
print('\tExecution time (s): \t\t', execution_time)
print('\tNumber of candidates examined: \t',numOfCandidates2,'\n')