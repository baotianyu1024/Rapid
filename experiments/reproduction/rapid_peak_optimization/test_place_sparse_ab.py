import ctypes,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p);rng=np.random.default_rng(12)
for g,nb in [(8,64),(64,8)]:
 for seed in range(4):
  n=g*40;nt=3;p=np.zeros((n,g),np.uint64);tables=rng.integers(0,nt,n,dtype=np.uint8)
  for i in range(n):
   ranks=[rng.integers(g)] if i<n*.8 else rng.choice(g,3,replace=False)
   for r in ranks:
    for b in rng.choice(nb,3,replace=False):p[i,r]|=np.uint64(1)<<np.uint64(b)
  initial=rng.integers(100,1000,(nb,nt,g),dtype=np.int64);outputs=[]
  for filename in ['incremental_place.so','incremental_place_sparse.so']:
   lib=ctypes.CDLL(str(root/filename));fn=lib.place_new;fn.argtypes=[ctypes.c_long]+[ctypes.c_int]*3+[ctypes.c_void_p]*6;fn.restype=ctypes.c_int;capacity=np.full(g,40,np.int64);owners=np.full(n,-1,np.int16);rd=initial.copy();sd=initial.copy();assert fn(n,g,nb,nt,*map(ptr,[tables,p,capacity,owners,rd,sd]))==0;outputs.append((owners,rd,sd))
  for a,b in zip(*outputs):assert np.array_equal(a,b)
print('SPARSE NEW-OWNER PLACEMENT EXACT AB PASS')
