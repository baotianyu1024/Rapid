import ctypes,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
lib=ctypes.CDLL(str(ROOT/'combined_owners.so'));fn=lib.refine;fn.restype=ctypes.c_long;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*10+[ctypes.c_double]*5+[ctypes.c_int]*2
ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
rng=np.random.default_rng(20261026)
checks=[]
for g,nb in [(8,8),(8,64),(64,8)]:
 for trial in range(8):
  n=300;present=rng.random((n,nb,g))<.12;owners=rng.integers(g,size=n,dtype=np.int16)
  def recount():
   recv=np.zeros((nb,g),dtype=np.int64);send=recv.copy()
   for i in range(n):
    remote=present[i].copy();remote[:,owners[i]]=False
    recv+=remote;send[:,owners[i]]+=remote.sum(1)
   return recv,send,np.bincount(owners,minlength=g).astype(np.int64)
  r,s,storage=recount();rd=r.copy();sd=s.copy();weight=np.ascontiguousarray(present.sum(2),dtype=np.uint16)
  profile=np.ascontiguousarray((present.astype(np.uint64)*(np.uint64(1)<<np.arange(nb,dtype=np.uint64))[None,:,None]).sum(1),dtype=np.uint64)
  total=r.sum(1).copy();norm=total.astype(float)/g
  def energy():
   z=norm[:,None]
   return (128*((r+s-2*z)**2/(4*z*z)).sum()+10*(r.sum(1)/norm).sum())/nb+10*((storage-storage.mean())**2/storage.mean()**2).sum()
  e0=energy()
  for epoch in range(3):
   fn(n,g,nb,*map(ptr,[owners,profile,weight,rd,sd,r,s,storage,norm,total]),128.,0.,10.,10.,.05,epoch%2,16)
   rr,ss,st=recount();assert np.array_equal(r,rr) and np.array_equal(s,ss) and np.array_equal(storage,st)
   assert np.array_equal(rd,rr) and np.array_equal(sd,ss)
   assert np.all(r.sum(1)<=total*1.05)
  assert energy()<=e0+1e-9
  checks.append(dict(g=g,nb=nb,trial=trial,energy_before=e0,energy_after=energy()))
(ROOT/'kernel_test.json').write_text(json.dumps(dict(success=True,cases=checks),indent=2))
print('PASS: 24 randomized exact counter, budget and combined-objective checks')
