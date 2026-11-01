from pathlib import Path
import sys,json,copy,time,ctypes,gc,hashlib
import numpy as np
V=Path(__file__).resolve().parents[1];H=V.parent;W=H.parents[1];X=H/'code/reproduction/rapid_peak_optimization';sys.path[:0]=[str(X),str(H/'code')]
from evidence_utils import save,read
from stream_owner_state import OwnerState
from stream_window import window
from stream_alternate import alternate
import group_scheduler as group
lib=ctypes.CDLL(str(X/'sample_group_peak_phase.so'));fn=lib.sample_group_peak;fn.argtypes=group.fn.argtypes;fn.restype=group.fn.restype;group.fn=fn
L=[3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1];g=8;b=8192;n=65536;N0=2097152;K=32;cfg=dict(batch_size=b,dense=13,table_lengths=L)
out=V/'results/long_incremental';out.mkdir(exist_ok=False);raw=np.memmap(W/'data/processed/criteo/day6.bin',dtype='<u4',mode='r').reshape(-1,228);base=np.ascontiguousarray(raw[:N0]);results=[]
def run_window(state,data,path,refit=True):
 path.mkdir();old=state.next_gid;t=time.perf_counter();first=path/'initial';window(state,data,cfg,first,rounds=4,balance=4,schedule='grouped');final=path/'final';alternate(state,first,data,final,rounds=4,balance=4,reallocate=refit,scheduler='group');return final,time.perf_counter()-t

def count(state,data,case,choice=None):
 assign=np.load(case/'assignment.npy');nb=len(data)//(g*b);assert len(assign)==len(data);r=np.zeros((nb,26,g),np.int64);s=r.copy();u=r.copy()
 for k in range(nb):
  a=assign[k*g*b:(k+1)*g*b];assert np.array_equal(np.bincount(a,minlength=g),np.full(g,b));x=data[k*g*b:(k+1)*g*b]
  for rank in range(g):
   records=x[a==rank];col=14
   for t,l in enumerate(L):
    ids=np.unique(records[:,col:col+l]);col+=l;own=state.owner[t][ids];assert np.all(own>=0);remote=own[own!=rank];r[k,t,rank]=len(remote);s[k,t]+=np.bincount(remote,minlength=g);u[k,t,rank]=len(ids)
 pred=np.load(case/'predicted_group_counts.npz');assert np.array_equal(pred['receive'],r.sum(1)) and np.array_equal(pred['send'],s.sum(1))
 choose=r.mean(axis=(0,2))<b*(g-1) if choice is None else choice
 fr=np.where(choose[None,:,None],r,b*(g-1)).sum(1);br=np.where(choose[None,:,None],s,b*(g-1)).sum(1)
 np.savez_compressed(case/'independent_counts.npz',receive=r,send=s,unique=u,fr=fr,br=br)
 return dict(vectors_one_epoch=int(fr.sum()*2),vectors_five_epochs=int(fr.sum()*10),vectors_per_8192=float(2*fr.mean()),mean_step_peak=float((fr+br).max(1).mean()),local_hit=float(1-r.sum()/u.sum()),fixed_acp=choose.tolist(),raw_unique_vectors=int(r.sum()*2),rows=len(data),audit='independent scan of original records equals group-kernel counts; exact per-batch capacities'),choose

def add_default(state,data):
 col=14;added=0
 for t,l in enumerate(L):
  ids=np.unique(data[:,col:col+l]);col+=l;ids=ids[state.lookup_owner(t,ids)<0]
  # Cyclic capacity-only admission; no affinity or access-frequency learning.
  owners=(np.arange(len(ids))+state.next_gid)%g;state.add_new(t,ids,owners);added+=len(ids)
 return added

def changes(previous,current):
 moved=slots=old=0
 for t in range(26):
  ids=np.flatnonzero(previous.owner[t]>=0);old+=len(ids);moved+=int(np.count_nonzero(previous.owner[t][ids]!=current.owner[t][ids]));slots+=int(np.count_nonzero(previous.physical[t][ids]!=current.physical[t][ids]))
 return dict(previous_embeddings=old,moved_embeddings=moved,physical_ids_changed=slots,weight_bytes=moved*512,adam_bytes=moved*1536,adam_plus_routes_bytes=moved*(1536+16*(g-1)),routing_assumption='16-byte records replicated to other7ranks; modeled payload,not transfer time',sample_reencoding_bytes=len(base)*912)

def fingerprint(state):
 return [(np.flatnonzero(o>=0),o[o>=0].copy(),p[o>=0].copy()) for o,p in zip(state.owner,state.physical)]
def check_old(state,snapshot):
 for t,(ids,o,p) in enumerate(snapshot):assert np.array_equal(state.owner[t][ids],o) and np.array_equal(state.physical[t][ids],p)
def summarize():
 save(out/'summary.json',dict(profile_rows=N0,window_rows=n,windows=K,epochs=5,ranks=g,batch=b,source='day6 shuffled file; disjoint sequential slices, NOT chronological drift',profile_algorithm='same phase-guarded window CPS planner for initial and full rebuild; no cache, no periodic tuning',full_checkpoints=checkpoints,results=results))
def pure_unique(data):
 r=np.zeros((1,26,g),np.int64);s=r.copy();u=r.copy()
 for rank in range(g):
  rec=data[rank*b:(rank+1)*b];col=14
  for t,l in enumerate(L):
   ids=np.unique(rec[:,col:col+l]);col+=l;own=ids%g;remote=own[own!=rank];r[0,t,rank]=len(remote);s[0,t]+=np.bincount(remote.astype(np.int64),minlength=g);u[0,t,rank]=len(ids)
 f=r.sum(1);q=s.sum(1);return dict(vectors_one_epoch=int(f.sum()*2),vectors_five_epochs=int(f.sum()*10),vectors_per_8192=float(2*f.mean()),mean_step_peak=float((f+q).max(1).mean()),local_hit=float(1-r.sum()/u.sum()))
initial=OwnerState(g,26);initialcase,sec=run_window(initial,base,out/'d0');met,choice=count(initial,base,initialcase);save(out/'initial.json',dict(metrics=met,seconds=sec));initial.save(out/'initial_state');checkpoints=[1,8,16,24,25,26,27,28,29,30,31,32]
for policy in ['incremental','frozen']:
 state=copy.deepcopy(initial)
 for k in range(1,K+1):
  data=np.ascontiguousarray(raw[N0+(k-1)*n:N0+k*n]);old=state.next_gid;before=fingerprint(state)
  if policy=='frozen':add_default(state,data)
  path,sec=run_window(state,data,out/f'{policy}_{k:02d}',refit=policy=='incremental');check_old(state,before);del before;met,_=count(state,data,path,choice)
  results.append(dict(policy=policy,window=k,metrics=met,new_embeddings=state.next_gid-old,known_embeddings=state.next_gid,old_owner_changes=0,old_physical_changes=0,seconds=sec))
  if policy=='incremental':results.append(dict(policy='pure_unique',window=k,metrics=pure_unique(data)))
  summarize();print('DONE',policy,k,met['vectors_per_8192'],flush=True)
 del state;gc.collect()
# Full rebuild on accumulated UNION+SHUFFLE, evaluate ONLY current original global batch.
for k in checkpoints:
 pool=np.ascontiguousarray(raw[:N0+k*n]);pool=pool[np.random.default_rng(20261026+k).permutation(len(pool))];state=OwnerState(g,26);path,sec=run_window(state,pool,out/f'full_profile_{k:02d}');del pool;data=np.ascontiguousarray(raw[N0+(k-1)*n:N0+k*n]);before=fingerprint(state);path,s2=run_window(state,data,out/f'full_rebuild_{k:02d}');check_old(state,before);del before;met,_=count(state,data,path,choice);results.append(dict(policy='full_rebuild',window=k,metrics=met,seconds=sec+s2));summarize();print('DONE full_rebuild',k,met['vectors_per_8192'],flush=True);del state;gc.collect()
x=np.array([next(r['metrics']['vectors_per_8192'] for r in results if r['policy']=='incremental' and r['window']==k) for k in range(1,K+1)]);last=x[-8:];prior=x[-16:-8];change=abs(last.mean()/prior.mean()-1);cv=last.std()/last.mean();slope=np.polyfit(np.arange(16),x[-16:],1)[0]*15/x[-16:].mean();stable=change<=.02 and cv<=.05 and abs(slope)<=.05
ratios=[next(r['metrics']['vectors_per_8192'] for r in results if r['policy']=='incremental' and r['window']==k)/next(r['metrics']['vectors_per_8192'] for r in results if r['policy']=='full_rebuild' and r['window']==k) for k in range(25,33)]
save(out/'stability.json',dict(criteria='predeclared: last8/prior8 mean change<=2%,last8 CV<=5%,16-window fitted end-to-end relative slope<=5%',mean_change=float(change),last8_cv=float(cv),relative_slope=float(slope),stable=bool(stable),last8_incremental_over_full=ratios,last8_mean_ratio=float(np.mean(ratios)),interpretation='tail ratios are called stable only if stable=true; no forced convergence claim'))
save(out/'complete.json',dict(success=True,stable=bool(stable)));print('COMPLETE',stable,flush=True)
