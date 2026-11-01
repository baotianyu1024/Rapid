"""Independent raw-record replay: actual occurrence counts, no profile cache used."""
import argparse,json,pickle,time
from pathlib import Path
import numpy as np
from persistent_replay import replay_window
p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);a=p.parse_args();exp=a.experiment;manifest=json.load(open(exp/'manifest.json'));metrics=json.load(open(exp/'metrics.json'));last=metrics[-1];case=exp/f"window_{last['window']:02d}";cfg=json.load(open(case/'config.json'));start=time.perf_counter()
with open(exp/'checkpoint.pkl','rb') as f:owner,_,completed=pickle.load(f)
assert completed==last['window']
# Independently audit actual final physical addresses, including free slots.
counts=np.zeros(owner.ranks,np.int64)
for t,o in enumerate(owner.owner):
 active=np.flatnonzero(o>=0);physical=owner.physical[t][active];assert np.array_equal(physical%owner.ranks,o[active]);unique=np.unique(physical);assert len(unique)==len(active);table_counts=np.bincount(o[active],minlength=owner.ranks);counts+=table_counts
 for r in range(owner.ranks):
  free=np.asarray(owner.free.get((t,r),[]),np.uint32);assert len(np.unique(free))==len(free);positions=np.searchsorted(unique,free);inside=positions<len(unique);assert not np.any(unique[positions[inside]]==free[inside]);assert np.all(free%owner.ranks==r)
  assert table_counts[r]+len(free)==owner.next[t,r]
assert np.array_equal(counts,owner.storage) and counts.sum()==owner.next_gid
with open(case/'incoming_cache.pkl','rb') as f:state=pickle.load(f)
assert np.array_equal(state.owner,owner.global_owners());g=cfg['ranks'];b=cfg['batch_size'];nb=cfg['batches'];n=cfg['rows'];lengths=cfg['table_lengths'];width=1+cfg['dense']+sum(lengths);data=np.memmap(manifest['source'],dtype='<u4',mode='r',offset=last['source_row_start']*width*4,shape=(n,width));assignment=np.load(case/'assignment.npy');trace=[]
for k in range(nb):
 batches=[]
 for r in range(g):
  rows=np.flatnonzero(assignment[k*g*b:(k+1)*g*b]==r)+k*g*b;assert len(rows)==b;record=data[rows];parts=[];counts=[];col=1+cfg['dense']
  for t,l in enumerate(lengths):
   fid,c=np.unique(record[:,col:col+l],return_counts=True);ids=owner.gid[t][fid];assert np.all(ids!=np.iinfo(np.uint32).max);parts.append(ids);counts.append(c);col+=l
  batches.append((np.concatenate(parts),np.concatenate(counts)))
 trace.append(batches);print('raw batch reconstructed',k,flush=True)
result,arrays=replay_window(state,trace);ref=np.load(case/'traffic.npz')
for key,actual in [('fr',arrays['forward'].sum(2)),('fs',arrays['forward'].sum(1)),('br',arrays['backward'].sum(1)),('bs',arrays['backward'].sum(2))]:assert np.array_equal(ref[key],actual),key
(case/'raw_verification.json').write_text(json.dumps(dict(success=True,original_records_used=True,final_physical_addresses_unique_and_owner_consistent=True,physical_free_slot_accounting_verified=True,actual_occurrence_counts_used=True,all_forward_backward_send_receive_arrays_identical=True,original_batch_capacity_verified=True,rows=n,steps=nb*5,seconds=time.perf_counter()-start),indent=2));print('RAW RECORD REPLAY VERIFIED',exp,flush=True)
