import numpy as np
from group_scheduler import grouped,ptr
rng=np.random.default_rng(92)
for g in [8,64]:
 nb=3;b=32;n=g*b*nb;nr=n//g;m=2000;width=12;keys=np.zeros(n,np.uint64)
 for k in range(nb):keys[k*b:k*b+16]=np.repeat(np.arange(1,9),2)
 bags=rng.integers(0,m,(9,8),dtype=np.uint32);data=np.empty((n,width),np.uint32);data[:,:8]=bags[keys];data[:,8:]=rng.integers(0,m,(n,4),dtype=np.uint32);data.sort(1);owners=rng.integers(0,g,m,dtype=np.int16);assignment=np.repeat(np.arange(g,dtype=np.int32),nr)
 def counts():
  f=np.zeros((nb,g),np.int64);s=f.copy()
  for k in range(nb):
   for r in range(g):
    ix=np.flatnonzero((assignment==r)&((np.arange(n)%nr)//b==k));assert len(ix)==b;ids=np.unique(data[ix]);remote=owners[ids][owners[ids]!=r];f[k,r]=len(remote);s[k]+=np.bincount(remote,minlength=g)
  return f,s
 f0,s0=counts();grouped(ptr(data),n,width,m,g,b,ptr(owners),ptr(assignment),8,4,keys);f,s=counts();assert (f+s).sum()<=(f0+s0).sum();assert np.all((f+s).max(1)<=(f0+s0).max(1));assert np.all(f.max(1)<=f0.max(1)) and np.all(s.max(1)<=s0.max(1))
print('GROUP EXCHANGE INDEPENDENT COUNTS AND CAPACITIES PASS')
