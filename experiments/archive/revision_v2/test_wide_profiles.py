from pathlib import Path
import sys,ctypes,json
import numpy as np
V=Path(__file__).resolve().parents[1];X=V.parent/'code/reproduction/rapid_peak_optimization';sys.path.insert(0,str(X))
import stream_window as old
import stream_tune
sys.path.insert(0,str(V/'code'))
import wide_stream_window as wide
lib=ctypes.CDLL(str(V/'code/wide_profiles.so'));st=lib.stats;st.argtypes=stream_tune.stats.argtypes;st.restype=None
ptr=old.ptr;rng=np.random.default_rng(42)
for nb in [1,8,32,64,65,96,129]:
 g=4;b=3;nf=8;n=nb*g*b;rec=np.zeros((n,3),np.uint32);rec[:,1:]=rng.integers(0,nf,(n,2));rec[:nf,1]=np.arange(nf);a=np.tile(np.repeat(np.arange(g,dtype=np.int32),b),nb);lut=np.arange(nf,dtype=np.int32);lp=np.array([lut.ctypes.data],np.uintp);sz=np.array([nf],np.int64);ls=np.array([2],np.int32);p=np.zeros((nf,g,(nb+63)//64),np.uint64)
 wide.profiles(ptr(rec),n,g,b,3,0,1,*map(ptr,[ls,lp,sz,a,p]));want=np.zeros_like(p)
 for i in range(n):
  for fid in rec[i,1:]:want[fid,a[i],(i//(g*b))//64]|=np.uint64(1)<<np.uint64((i//(g*b))%64)
 assert np.array_equal(p,want)
 own=rng.integers(0,g,nf,dtype=np.int16);weights=np.zeros((nf,nb),np.uint16);r=np.zeros((nb,g),np.int64);s=r.copy();st(nf,g,nb,*map(ptr,[p,own,weights,r,s]));rr=np.zeros_like(r);ss=rr.copy()
 for batch in range(nb):
  for rank in range(g):
   ids=np.unique(rec[batch*g*b+rank*b:batch*g*b+(rank+1)*b,1:]);remote=own[ids];remote=remote[remote!=rank];rr[batch,rank]=len(remote);ss[batch]+=np.bincount(remote,minlength=g)
 assert np.array_equal(r,rr) and np.array_equal(s,ss)
 tabs=np.zeros(nf,np.uint8);cap=np.full(g,nf//g,np.int64);o=np.full(nf,-1,np.int16);r=np.zeros((nb,1,g),np.int64);s=r.copy();assert wide.place(nf,g,nb,1,*map(ptr,[tabs,p,cap,o,r,s]))==0
 rr=np.zeros((nb,g),np.int64);ss=rr.copy()
 for batch in range(nb):
  for rank in range(g):
   ids=np.unique(rec[batch*g*b+rank*b:batch*g*b+(rank+1)*b,1:]);remote=o[ids];remote=remote[remote!=rank];rr[batch,rank]=len(remote);ss[batch]+=np.bincount(remote,minlength=g)
 assert np.array_equal(r[:,0],rr) and np.array_equal(s[:,0],ss)
 if nb<=64:
  cap0=np.full(g,nf//g,np.int64);o0=np.full(nf,-1,np.int16);r0=np.zeros_like(r);s0=r0.copy();assert old.place(nf,g,nb,1,*map(ptr,[tabs,p,cap0,o0,r0,s0]))==0;assert np.array_equal(o,o0) and np.array_equal(r,r0) and np.array_equal(s,s0)
 print('PASS',nb,flush=True)
(V/'results/wide_profiles_validation.json').write_text(json.dumps(dict(batch_counts=[1,8,32,64,65,96,129],independent_profile_and_stats=True,original_placement_exact_equal_up_to64=True)))
