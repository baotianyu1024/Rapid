"""New-window-only placement and scheduling; no existing embedding can move."""
import ctypes,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/hugui/longjiu.bty/reproduction/revision_response_20261026/code/reproduction/rapid_peak_optimization')
ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
lib=ctypes.CDLL(str(ROOT/'incremental_place.so'))
initial=lib.initial_schedule;initial.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*6+[ctypes.c_void_p]*3+[ctypes.c_int]+[ctypes.c_void_p]*3;initial.restype=None
initial_random=ctypes.CDLL(str(ROOT/'incremental_place_random.so')).initial_schedule;initial_random.argtypes=initial.argtypes;initial_random.restype=None
initial_grouped=ctypes.CDLL(str(ROOT/'incremental_place_grouped.so')).initial_schedule;initial_grouped.argtypes=initial.argtypes;initial_grouped.restype=None
profiles=lib.new_profiles;profiles.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*6+[ctypes.c_void_p]*5;profiles.restype=None
place=lib.place_new;place.argtypes=[ctypes.c_long]+[ctypes.c_int]*3+[ctypes.c_void_p]*6;place.restype=ctypes.c_int
place_sparse=ctypes.CDLL(str(ROOT/'incremental_place_sparse.so')).place_new;place_sparse.argtypes=place.argtypes;place_sparse.restype=place.restype;place=place_sparse
peaklib=ctypes.CDLL(str(ROOT/'batch_peak_samples.so'));peak=peaklib.batch_peak_samples;peak.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]*2+[ctypes.c_int,ctypes.c_double];peak.restype=ctypes.c_int

wide=ctypes.CDLL('/mnt/hugui/longjiu.bty/reproduction/revision_response_20261026/zh_revision_v2/code/wide_profiles.so')
for name in ['new_profiles','place_new']:
 old=profiles if name=='new_profiles' else place
 fn=getattr(wide,name);fn.argtypes=old.argtypes;fn.restype=old.restype
 if name=='new_profiles':profiles=fn
 else:place=fn

def frozen_hash(state):
 h=hashlib.sha256()
 for o,p in zip(state.owner,state.physical):
  known=o>=0;h.update(np.flatnonzero(known).astype(np.uint32).tobytes());h.update(o[known].tobytes());h.update(p[known].tobytes())
 return h.hexdigest()

def window(state,data,cfg,out,rounds=4,balance=4,schedule="affinity"):
 out=Path(out);out.mkdir(exist_ok=False);cache=out/'profile_cache';cache.mkdir();start=time.perf_counter();g=state.ranks;n=len(data);b=cfg['batch_size'];nb=n//(g*b);lengths=np.array(cfg['table_lengths'],np.int32);nt=len(lengths);width=data.shape[1];dense=cfg['dense'];assert n==nb*g*b and 0<nb<=4096
 cfg={k:cfg[k] for k in ['batch_size','dense','table_lengths']};cfg.update(ranks=g,rows=n,batches=nb,initial_sample_schedule=schedule,global_batch_membership_preserved=True);(out/'config.json').write_text(json.dumps(cfg,indent=2))
 old_n=state.next_gid;before=[]
 for t in range(nt):
  ix=np.flatnonzero(state.owner[t]>=0);before.append((ix,state.owner[t][ix].copy(),state.physical[t][ix].copy()))
 allids=[];freq=[];newids=[];top=[];col=dense+1
 for t,l in enumerate(lengths):
  ids,c=np.unique(data[:,col:col+l],return_counts=True);allids.append(ids);freq.append(c);newids.append(ids[state.lookup_owner(t,ids)<0]);k=min(40000,len(ids));ix=np.argpartition(c,len(c)-k)[-k:];top.extend((int(c[i]),t,int(ids[i])) for i in ix);state._grow(t,int(ids.max())+1);col+=l
  print('discover',t,'new',len(newids[-1]),'sec',round(time.perf_counter()-start,1),flush=True)
 hot=sorted(top,key=lambda x:(-x[0],x[1],x[2]))[:40000];hotsets=[set() for _ in range(nt)]
 for c,t,f in hot:
  if state.owner[t][f]<0:hotsets[t].add(f)
 total_new=sum(map(len,newids));target=np.full(g,(old_n+total_new)//g,np.int64);target[:(old_n+total_new)%g]+=1;cap=target-state.storage;assert np.all(cap>=0)
 rng=np.random.default_rng(12345);starts=rng.integers(0,g,nt);cold=[];hotcount=0
 for t in range(nt):
  hi=np.array(sorted(hotsets[t]),np.uint32);ho=[];r=int(starts[t])
  for fid in hi:
   while cap[r]==0:r=(r+1)%g
   ho.append(r);cap[r]-=1;r=(r+1)%g
  np.save(out/f'{t}.new_hot_ids.npy',hi);state.add_new(t,hi,np.array(ho,np.int16));hotcount+=len(hi);cold.append(np.setdiff1d(newids[t],hi,assume_unique=True))
 op=np.array([x.ctypes.data for x in state.owner],np.uintp);sizes=np.array([len(x) for x in state.owner],np.int64);assignment=np.zeros(n,np.int32);rd=np.zeros((nb,nt,g),np.int64);sd=rd.copy()
 (initial_grouped if schedule=="grouped" else initial_random if schedule=="random" else initial)(ptr(data),n,g,b,width,dense,nt,*map(ptr,[lengths,op,sizes]),42,*map(ptr,[assignment,rd,sd]));print('initial schedule',round(time.perf_counter()-start,1),flush=True)
 lookup=[];tables=[];offset=0
 for t,ids in enumerate(cold):
  lut=np.full(len(state.owner[t]),-1,np.int32);lut[ids]=np.arange(offset,offset+len(ids));lookup.append(lut);tables.append(np.full(len(ids),t,np.uint8));offset+=len(ids)
 lp=np.array([x.ctypes.data for x in lookup],np.uintp);pr=np.zeros((offset,g,(nb+63)//64),np.uint64);profiles(ptr(data),n,g,b,width,dense,nt,*map(ptr,[lengths,lp,sizes,assignment,pr]));newown=np.full(offset,-1,np.int16);tabs=np.concatenate(tables)
 assert place(offset,g,nb,nt,*map(ptr,[tabs,pr,cap,newown,rd,sd]))==0 and not cap.any();off=0
 for t,ids in enumerate(cold):state.add_new(t,ids,newown[off:off+len(ids)]);off+=len(ids)
 del lookup,lp,pr,newown,tabs;assert np.array_equal(state.storage,target)
 for t,(ix,o,p) in enumerate(before):assert np.array_equal(state.owner[t][ix],o) and np.array_equal(state.physical[t][ix],p)
 del before
 # Rank-major permutation preserves original global batch identity by construction.
 order=np.argsort(assignment,kind='stable');nr=n//g;batchid=np.arange(n)//(g*b)
 assert np.array_equal(batchid[order],np.tile(np.repeat(np.arange(nb),b),g))
 compact=np.empty((n,int(lengths.sum())),np.uint32);owners=[];off=0;col=dense+1;dst=0
 for t,(ids,l) in enumerate(zip(allids,lengths)):
  lut=np.full(len(state.owner[t]),np.iinfo(np.uint32).max,np.uint32);lut[ids]=np.arange(off,off+len(ids),dtype=np.uint32);compact[:,dst:dst+l]=lut[data[order,col:col+l]];owners.append(state.owner[t][ids]);off+=len(ids);col+=l;dst+=l
 compact.sort(axis=1);own=np.concatenate(owners);assign=np.repeat(np.arange(g,dtype=np.int32),nr);moves=peak(ptr(compact),n,int(lengths.sum()),off,g,b,ptr(own),ptr(assign),rounds,balance);assignment[order]=assign;del compact,order,own,assign
 for k in range(nb):assert np.array_equal(np.bincount(assignment[k*g*b:(k+1)*g*b],minlength=g),np.full(g,b))
 np.save(out/'assignment.npy',assignment);np.savez_compressed(out/'initial_counts.npz',receive=rd,send=sd)
 # Independent per-table profiles from original sample records and final assignments.
 for t,ids in enumerate(allids):
  lut=np.full(len(state.owner[t]),-1,np.int32);lut[ids]=np.arange(len(ids));lptr=np.array([lut.ctypes.data],np.uintp);sz=np.array([len(lut)],np.int64);ls=np.array([lengths[t]],np.int32);col=dense+1+int(lengths[:t].sum())
  # The scanner sees a compact record containing this table and no dense columns.
  rec=np.empty((n,1+int(lengths[t])),np.uint32);rec[:,0]=0;rec[:,1:]=data[:,col:col+lengths[t]];pr=np.zeros((len(ids),g,(nb+63)//64),np.uint64)
  profiles(ptr(rec),n,g,b,rec.shape[1],0,1,*map(ptr,[ls,lptr,sz,assignment,pr]));np.save(cache/f'{t}.profiles.npy',pr.astype(np.uint8 if nb<=8 else np.uint16 if nb<=16 else np.uint32 if nb<=32 else np.uint64));np.save(cache/f'{t}.ids.npy',state.physical[t][ids]);np.save(out/f'{t}.original_ids.npy',ids);np.save(out/f'{t}.owners.npy',state.owner[t][ids]);np.save(out/f'{t}.logical_gids.npy',state.gid[t][ids]);print('profiles',t,flush=True)
 result=dict(rows=n,old_embeddings=old_n,new_embeddings=total_new,new_hot_embeddings=hotcount,known_embeddings=state.next_gid,old_owners_and_physical_ids_unchanged=True,global_batch_membership_preserved=True,rank_samples_per_batch=b,storage=state.storage.tolist(),sample_refinement_moves=moves,seconds=time.perf_counter()-start)
 (out/'placement.json').write_text(json.dumps(result,indent=2));return result
