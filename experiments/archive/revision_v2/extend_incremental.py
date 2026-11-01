from pathlib import Path
import sys,json,copy,time,ctypes,gc
import numpy as np
V=Path(__file__).resolve().parents[1];H=V.parent;W=H.parents[1];X=H/'code/reproduction/rapid_peak_optimization';sys.path[:0]=[str(X),str(H/'code'),str(V/'code')]
# Reuse the exact counting/admission functions; do not execute the initial run.
src=(V/'code/long_incremental.py').read_text().split('initial=OwnerState(g,26);')[0]
src=src.replace("out.mkdir(exist_ok=False)","out.mkdir(exist_ok=True)")
exec(compile(src,str(V/'code/long_incremental.py'),'exec'))
original=out;out=V/'results/incremental_extension';out.mkdir(exist_ok=False)
from wide_stream_window import window
from wide_stream_alternate import alternate
initial=OwnerState.from_export(original/'initial_state');choice=np.array(read(original/'initial.json')['metrics']['fixed_acp']);results=[r for r in read(original/'summary.json')['results'] if r['policy'] in ['incremental','pure_unique']];checkpoints=[]
def recover(policy):
 state=copy.deepcopy(initial)
 for k in range(1,33):
  p=original/f'{policy}_{k:02d}'/'final'
  for t in range(26):
   ids=np.load(p/f'{t}.original_ids.npy');own=np.load(p/f'{t}.owners.npy');phys=np.load(p/f'profile_cache/{t}.ids.npy');gids=np.load(p/f'{t}.logical_gids.npy');state._grow(t,int(ids.max())+1);known=state.owner[t][ids]>=0
   assert np.array_equal(state.owner[t][ids[known]],own[known]) and np.array_equal(state.physical[t][ids[known]],phys[known]);state.owner[t][ids]=own;state.physical[t][ids]=phys;state.gid[t][ids]=gids
 state.storage[:]=0;state.free={};state.next_gid=0
 for t in range(26):
  ids=np.flatnonzero(state.owner[t]>=0);state.next_gid+=len(ids);state.storage+=np.bincount(state.owner[t][ids],minlength=g)
  for r in range(g):
   phys=state.physical[t][ids[state.owner[t][ids]==r]];nextslot=int((phys//g).max())+1 if len(phys) else 0;state.next[t,r]=nextslot;state.free[(t,r)]=np.setdiff1d(np.arange(nextslot,dtype=np.uint32)*g+r,phys).tolist()
 assert np.array_equal(np.sort(np.concatenate([state.gid[t][state.owner[t]>=0] for t in range(26)])),np.arange(state.next_gid));state.audit();return state

def stability(k):
 x=np.array([next(r['metrics']['vectors_per_8192'] for r in results if r['policy']=='incremental' and r['window']==i) for i in range(1,k+1)]);last=x[-8:];prior=x[-16:-8];change=abs(last.mean()/prior.mean()-1);cv=last.std()/last.mean();slope=np.polyfit(np.arange(16),x[-16:],1)[0]*15/x[-16:].mean();return dict(windows=k,mean_change=float(change),last8_cv=float(cv),relative_slope=float(slope),stable=bool(change<=.02 and cv<=.05 and abs(slope)<=.05),criteria='last8/prior8<=2%,last8CV<=5%,16-window fitted relative slope<=5%; inspected at40,48,56,64')
state=recover('incremental');state.save(out/'recovered_incremental32');tests=[]
for k in range(33,65):
 data=np.ascontiguousarray(raw[N0+(k-1)*n:N0+k*n]);old=state.next_gid;before=fingerprint(state);path,sec=run_window(state,data,out/f'incremental_{k:02d}');check_old(state,before);del before;met,_=count(state,data,path,choice);results.append(dict(policy='incremental',window=k,metrics=met,new_embeddings=state.next_gid-old,known_embeddings=state.next_gid,old_owner_changes=0,old_physical_changes=0,seconds=sec));results.append(dict(policy='pure_unique',window=k,metrics=pure_unique(data)));K=k;summarize();print('DONE incremental',k,met['vectors_per_8192'],flush=True)
 if k%8==0:
  test=stability(k);tests.append(test);save(out/'stability_checks.json',tests)
  if test['stable']:break
state.save(out/'final_incremental_state');del state;gc.collect();K=k;checkpoints=list(range(K-7,K+1));summarize()
# The first run may still be finishing frozen policy; wait only for required files.
while not (original/'frozen_32/final/placement.json').exists():time.sleep(10)
results.extend(r for r in read(original/'summary.json')['results'] if r['policy']=='frozen');state=recover('frozen')
for k in range(33,K+1):
 data=np.ascontiguousarray(raw[N0+(k-1)*n:N0+k*n]);old=state.next_gid;before=fingerprint(state);add_default(state,data);path,sec=run_window(state,data,out/f'frozen_{k:02d}',refit=False);check_old(state,before);del before;met,_=count(state,data,path,choice);results.append(dict(policy='frozen',window=k,metrics=met,new_embeddings=state.next_gid-old,known_embeddings=state.next_gid,old_owner_changes=0,old_physical_changes=0,seconds=sec));summarize();print('DONE frozen',k,met['vectors_per_8192'],flush=True)
state.save(out/'final_frozen_state');del state;gc.collect()
for k in checkpoints:
 pool=np.ascontiguousarray(raw[:N0+k*n]);pool=pool[np.random.default_rng(20261026+k).permutation(len(pool))];state=OwnerState(g,26);path,sec=run_window(state,pool,out/f'full_profile_{k:02d}');del pool;data=np.ascontiguousarray(raw[N0+(k-1)*n:N0+k*n]);before=fingerprint(state);path,s2=run_window(state,data,out/f'full_rebuild_{k:02d}');check_old(state,before);del before;met,_=count(state,data,path,choice);results.append(dict(policy='full_rebuild',window=k,metrics=met,seconds=sec+s2));summarize();print('DONE full_rebuild',k,met['vectors_per_8192'],flush=True);del state;gc.collect()
results.extend(r for r in read(original/'summary.json')['results'] if r['policy']=='full_rebuild' and r['window'] not in checkpoints);checkpoints=sorted({r['window'] for r in results if r['policy']=='full_rebuild'});summarize()
test=stability(K);ratios=[next(r['metrics']['vectors_per_8192'] for r in results if r['policy']=='incremental' and r['window']==k)/next(r['metrics']['vectors_per_8192'] for r in results if r['policy']=='full_rebuild' and r['window']==k) for k in range(K-7,K+1)];test.update(last8_incremental_over_full=ratios,last8_mean_ratio=float(np.mean(ratios)),tail_start=K-7,tail_end=K);save(out/'stability.json',test);save(out/'complete.json',dict(success=True,stable=test['stable']));print('COMPLETE',test,flush=True)
