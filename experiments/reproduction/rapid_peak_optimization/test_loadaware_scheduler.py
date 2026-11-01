import ctypes
import numpy as np
from pathlib import Path
from loadaware_scheduler import loadaware as fn,ptr
rng=np.random.default_rng(82)
for g in [8,64]:
 nb=3;b=16;n=g*b*nb;nr=n//g;m=g*8;width=12;data=np.sort(rng.integers(0,m,(n,width),dtype=np.uint32),axis=1);owner=rng.integers(0,g,m,dtype=np.int16);assign=np.repeat(np.arange(g,dtype=np.int32),nr)
 def metrics(a):
  f=np.zeros((nb,g),np.int64);s=f.copy()
  for k in range(nb):
   for r in range(g):
    ix=np.flatnonzero((a==r)&((np.arange(n)%nr)//b==k));assert len(ix)==b;ids=np.unique(data[ix]);remote=owner[ids][owner[ids]!=r];f[k,r]=len(remote);s[k]+=np.bincount(remote,minlength=g)
  return (f+s).mean(),(f+s).max(1).mean()
 before=metrics(assign);assert fn(ptr(data),n,width,m,g,b,ptr(owner),ptr(assign),3,4)>=0;after=metrics(assign);assert after[0]<=before[0] and after[1]<=before[1]
print('CAPACITATED BATCH SCHEDULER PASS')
