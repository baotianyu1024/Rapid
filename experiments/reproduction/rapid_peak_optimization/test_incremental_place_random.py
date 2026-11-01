import ctypes,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;lib=ctypes.CDLL(str(root/'incremental_place_random.so'));initial=lib.initial_schedule;initial.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*6+[ctypes.c_void_p]*3+[ctypes.c_int]+[ctypes.c_void_p]*3;initial.restype=None
profiles=lib.new_profiles;profiles.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*6+[ctypes.c_void_p]*5;profiles.restype=None
place=lib.place_new;place.argtypes=[ctypes.c_long]+[ctypes.c_int]*3+[ctypes.c_void_p]*6;place.restype=ctypes.c_int
ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p);rng=np.random.default_rng(8)
for g in [8,64]:
 b=16;nb=3;nt=3;lengths=np.array([3,2,1],np.int32);width=8;n=g*b*nb;m=g*4;data=rng.integers(0,m,(n,width),dtype=np.uint32);owners=[np.r_[np.arange(m//2)%g,np.full(m//2,-1)].astype(np.int16) for _ in range(nt)];old=[x.copy() for x in owners];sizes=np.array([m]*nt,np.int64);op=np.array([x.ctypes.data for x in owners],np.uintp);assign=np.zeros(n,np.int32);rd=np.zeros((nb,nt,g),np.int64);sd=rd.copy()
 initial(ptr(data),n,g,b,width,1,nt,*map(ptr,[lengths,op,sizes]),42,*map(ptr,[assign,rd,sd]))
 for k in range(nb):assert np.array_equal(np.bincount(assign[k*g*b:(k+1)*g*b],minlength=g),np.full(g,b))
 def count():
  rr=np.zeros_like(rd);ss=rr.copy();off=2
  for t,l in enumerate(lengths):
   for k in range(nb):
    for r in range(g):
     rows=np.flatnonzero(assign[k*g*b:(k+1)*g*b]==r)+k*g*b;ids=np.unique(data[rows,off:off+l]);o=owners[t][ids];remote=o[(o>=0)&(o!=r)];rr[k,t,r]=len(remote);ss[k,t]+=np.bincount(remote,minlength=g)
   off+=l
  return rr,ss
 rr,ss=count();assert np.array_equal(rr,rd) and np.array_equal(ss,sd)
 lookup=[];newids=[];tables=[];offset=0;col=2
 for t,l in enumerate(lengths):
  ids=np.unique(data[:,col:col+l]);ids=ids[owners[t][ids]<0];newids.append(ids);lut=np.full(m,-1,np.int32);lut[ids]=np.arange(offset,offset+len(ids));lookup.append(lut);tables.extend([t]*len(ids));offset+=len(ids);col+=l
 lp=np.array([x.ctypes.data for x in lookup],np.uintp);pr=np.zeros((offset,g),np.uint64);profiles(ptr(data),n,g,b,width,1,nt,*map(ptr,[lengths,lp,sizes,assign,pr]))
 expected=np.zeros_like(pr);col=2
 for t,l in enumerate(lengths):
  for i in range(n):
   for fid in np.unique(data[i,col:col+l]):
    ix=lookup[t][fid]
    if ix>=0:expected[ix,assign[i]]|=np.uint64(1)<<np.uint64(i//(g*b))
  col+=l
 assert np.array_equal(pr,expected)
 storage=sum((np.bincount(x[x>=0],minlength=g) for x in owners),np.zeros(g,np.int64));total=int(storage.sum()+offset);target=np.full(g,total//g,np.int64);target[:total%g]+=1;capacity=target-storage;out=np.full(offset,-1,np.int16);tables=np.array(tables,np.uint8)
 assert place(offset,g,nb,nt,*map(ptr,[tables,pr,capacity,out,rd,sd]))==0;assert not capacity.any();off=0
 for t,ids in enumerate(newids):owners[t][ids]=out[off:off+len(ids)];off+=len(ids);assert np.array_equal(owners[t][old[t]>=0],old[t][old[t]>=0])
 rr,ss=count();assert np.array_equal(rr,rd) and np.array_equal(ss,sd);assert np.array_equal(storage+np.bincount(out,minlength=g),target)
(root/'incremental_place_random_test.json').write_text(json.dumps(dict(success=True,ranks=[8,64],checks=['exact original-batch sample capacities','independent existing and new-ID traffic counts','exact new-ID rank/batch presence profiles','all old owners frozen','global storage count targets']),indent=2));print('INCREMENTAL PLACEMENT PASS')
