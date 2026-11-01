"""Strict-s0 cache replay with stable logical IDs across incremental windows."""
import ctypes,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'het_gmp_rapid/scripts'))
from replay_het import State
ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)

def update_owners(state,owners,changed_gids=()):
 owners=np.asarray(owners,dtype=np.int16);old_n=len(state.owner);changed=np.sort(np.asarray(changed_gids,dtype=np.int64));assert state.bound==0 and len(owners)>=old_n and np.all((owners>=0)&(owners<state.p))
 assert np.array_equal(np.flatnonzero(state.owner!=owners[:old_n]),changed)
 invalidated=[]
 for r,ids in enumerate(state.mirrors):
  assert not state.pending[r].any();ix=np.searchsorted(ids,changed);ok=ix<len(ids)
  if len(ids):ok&=ids[np.minimum(ix,len(ids)-1)]==changed
  pos=ix[ok];state.version[r][pos]=-1;invalidated.append(len(pos))
 if len(owners)>old_n:state.master=np.pad(state.master,(0,len(owners)-old_n),constant_values=1)
 state.owner=owners
 return invalidated

def set_mirrors(state,mirrors):
 assert len(mirrors)==state.p
 versions=[]
 for r,ids in enumerate(mirrors):
  ids=np.asarray(ids,dtype=np.uint32);assert np.all(np.diff(ids.astype(np.int64))>0) and np.all(state.owner[ids]!=r)
  _,oldidx,newidx=np.intersect1d(state.mirrors[r],ids,assume_unique=True,return_indices=True);v=np.full(len(ids),-1,np.int64);v[newidx]=state.version[r][oldidx];versions.append(v)
 state.mirrors=[np.asarray(x,dtype=np.uint32) for x in mirrors];state.version=versions;state.pending=[np.zeros(len(x),np.int64) for x in mirrors]

def trace_from_case(case,owner_state):
 case=Path(case);cfg=json.load(open(case/'config.json'));g=cfg['ranks'];nb=cfg['batches'];assert g<=64 and 1<=nb<=64;parts=[[] for _ in range(g*nb)];lib=ctypes.CDLL(str(ROOT/'profile_trace.so'));fn=lib.build_trace;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;fn.restype=ctypes.c_void_p;release=lib.free_trace;release.argtypes=[ctypes.c_void_p];release.restype=None
 for t in range(len(cfg['table_lengths'])):
  ids=np.load(case/f'profile_cache/{t}.ids.npy');gids=owner_state.profile_gids(t,ids);np.save(case/f'{t}.logical_gids.npy',gids);prof=np.asarray(np.load(case/f'profile_cache/{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64);counts=np.zeros(nb*g,np.int64);raw=fn(len(ids),g,nb,*map(ptr,[prof,gids,counts]));assert raw
  try:flat=np.ctypeslib.as_array(ctypes.cast(raw,ctypes.POINTER(ctypes.c_uint32)),shape=(int(counts.sum()),)).copy()
  finally:release(raw)
  offsets=np.r_[0,counts.cumsum()]
  for j in range(g*nb):parts[j].append(flat[offsets[j]:offsets[j+1]])
  print('logical trace table',t,flush=True)
 result=[]
 for b in range(nb):
  batches=[]
  for r in range(g):
   ids=np.concatenate(parts[b*g+r]);batches.append((ids,np.ones(len(ids),np.uint16)))
  result.append(batches)
 return result

def select_mirrors(case,state,budgets,out):
 case=Path(case);out=Path(out);out.mkdir(exist_ok=False);cfg=json.load(open(case/'config.json'));g=cfg['ranks'];nb=cfg['batches'];budgets=np.asarray(budgets,dtype=int);assert len(budgets)==g and state.bound==0
 lib=ctypes.CDLL(str(ROOT/'stream_cache_scores.so'));fn=lib.scores;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;fn.restype=None
 selected=[(np.empty(0,np.uint32),np.empty(0,np.uint16)) for _ in range(g)];valid=[ids[state.version[r]==state.master[ids]] for r,ids in enumerate(state.mirrors)];lookup=np.full(len(state.owner),np.iinfo(np.uint32).max,np.uint32);start=time.perf_counter()
 for t in range(len(cfg['table_lengths'])):
  gids=np.load(case/f'{t}.logical_gids.npy');prof=np.asarray(np.load(case/f'profile_cache/{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64);own=np.asarray(state.owner[gids],dtype=np.int16);n=len(gids);assert np.array_equal(own,np.load(case/f'{t}.owners.npy'));scores=np.zeros((g,n),np.uint16);fn(n,g,nb,*map(ptr,[prof,own,scores]));lookup[gids]=np.arange(n,dtype=np.uint32);allbits=np.bitwise_or.reduce(prof,axis=1);first=allbits&(np.bitwise_not(allbits)+np.uint64(1))
  for r in range(g):
   ix=lookup[valid[r]];ix=ix[ix!=np.iinfo(np.uint32).max];ix=ix[(prof[ix,r]&first[ix])!=0];assert np.all(own[ix]!=r);scores[r,ix]+=1
   ix=np.flatnonzero(scores[r]);ids=np.r_[selected[r][0],gids[ix]];sc=np.r_[selected[r][1],scores[r,ix]]
   if len(ids)>budgets[r]:
    cut=np.partition(sc,len(sc)-budgets[r])[len(sc)-budgets[r]];hi=np.flatnonzero(sc>cut);eq=np.flatnonzero(sc==cut);need=budgets[r]-len(hi)
    if need<len(eq):eq=eq[np.argpartition(ids[eq],need)[:need]] if need else eq[:0]
    keep=np.r_[hi,eq];ids=ids[keep];sc=sc[keep]
   order=np.argsort(ids);selected[r]=(ids[order],sc[order])
  lookup[gids]=np.iinfo(np.uint32).max;print('persistent mirror selection table',t,flush=True)
 mirrors=[];expected=[];retained=[]
 for r,(ids,sc) in enumerate(selected):
  expected.append(int(sc.sum()));spare=int(budgets[r]-len(ids));extra=np.empty(0,np.uint32)
  if spare:
   extras=np.setdiff1d(state.mirrors[r],ids,assume_unique=True);extras=extras[state.owner[extras]!=r];positions=np.searchsorted(state.mirrors[r],extras);fresh=state.version[r][positions]==state.master[extras];extra=np.r_[extras[fresh],extras[~fresh]][:spare]
  retained.append(len(extra));ids=np.sort(np.r_[ids,extra]).astype(np.uint32);assert len(ids)<=budgets[r] and np.all(state.owner[ids]!=r);mirrors.append(ids);np.save(out/f'mirrors_{r}.npy',ids)
 (out/'selection.json').write_text(json.dumps(dict(expected_saved_vectors_by_rank=expected,mirror_capacity=budgets.tolist(),selected_counts=list(map(len,mirrors)),zero_current_benefit_retained_counts=retained,initial_warm_versions_used=True,seconds=time.perf_counter()-start),indent=2));return mirrors,np.array(expected,np.int64)

def replay_window(state,trace,epochs=5,expected_saved=None):
 assert not state.history;raw=np.zeros(state.p,np.int64);start=time.perf_counter();prepared=[]
 for batches in trace:
  prepared.append(state.prepare(batches))
  for r,(ids,counts) in enumerate(batches):raw[r]+=int((state.owner[ids]!=r).sum())
 for epoch in range(epochs):
  for batches,prep in zip(trace,prepared):state.step(batches,prep)
 result,arrays=state.finish();saved=raw*epochs-arrays['forward'].sum((0,2))
 if expected_saved is not None:assert np.array_equal(saved,expected_saved),(saved,expected_saved)
 result.update(persistent_cache=True,update_counts_normalized_to_presence=True,epochs=epochs,steps_per_epoch=len(trace),saved_forward_vectors_by_rank=saved.tolist(),replay_seconds=time.perf_counter()-start)
 state.history=[];return result,arrays
