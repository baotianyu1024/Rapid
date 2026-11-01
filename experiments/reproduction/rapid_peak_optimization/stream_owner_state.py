"""Logical embedding ownership with stable physical IDs between checkpoints.
This models placement/migration, not actual GPU tensor transport.
"""
from pathlib import Path
import json
import numpy as np
from migration_accounting import migration_ledger

class OwnerState:
 def __init__(self,ranks,tables):
  assert 1<=ranks<=64 and tables>0
  self.ranks=ranks;self.owner=[np.full(0,-1,np.int16) for _ in range(tables)];self.physical=[np.empty(0,np.uint32) for _ in range(tables)];self.gid=[np.empty(0,np.uint32) for _ in range(tables)];self.next_gid=0;self.next=np.zeros((tables,ranks),np.int64);self.free={};self.storage=np.zeros(ranks,np.int64)
 @classmethod
 def from_export(cls,path):
  path=Path(path);meta=json.load(open(path/'manifest.json'));files=sorted(path.glob('table_*.npz'));state=cls(meta['ranks'],len(files))
  for t in range(len(files)):
   z=np.load(path/f'table_{t}.npz');ids=z['original_ids'].astype(np.int64);physical=z['physical_ids'];own=z['owners'];state._grow(t,int(ids.max())+1);state.owner[t][ids]=own;state.physical[t][ids]=physical
   state.gid[t][ids]=z['logical_gids'] if 'logical_gids' in z else np.arange(state.next_gid,state.next_gid+len(ids),dtype=np.uint32);state.next_gid+=len(ids)
   assert np.array_equal(physical%state.ranks,own) and len(np.unique(physical))==len(physical)
   for r in range(state.ranks):
    p=physical[own==r];state.next[t,r]=int((p//state.ranks).max())+1 if len(p) else 0
   state.storage+=np.bincount(own,minlength=state.ranks)
  assert np.array_equal(state.storage,meta['primary_counts']);state.next_gid=meta.get('next_gid',state.next_gid)
  if 'next_slots' in meta:state.next=np.array(meta['next_slots'],np.int64)
  state.free={tuple(map(int,k.split(','))):v for k,v in meta.get('free_slots',{}).items()};return state
 def _grow(self,t,length):
  if length>len(self.owner[t]):
   old=len(self.owner[t]);self.owner[t]=np.pad(self.owner[t],(0,length-old),constant_values=-1);self.physical[t]=np.pad(self.physical[t],(0,length-old),constant_values=0);self.gid[t]=np.pad(self.gid[t],(0,length-old),constant_values=np.iinfo(np.uint32).max)
 def lookup_owner(self,t,ids):
  ids=np.asarray(ids,dtype=np.int64);out=np.full(ids.shape,-1,np.int16);valid=(ids>=0)&(ids<len(self.owner[t]));out[valid]=self.owner[t][ids[valid]];return out
 def add_new(self,t,ids,owners):
  ids=np.asarray(ids,dtype=np.int64);owners=np.asarray(owners,dtype=np.int64)
  if ids.shape!=owners.shape or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any((owners<0)|(owners>=self.ranks)):raise ValueError('Invalid new-ID assignment')
  if np.any(self.lookup_owner(t,ids)>=0):raise ValueError('Existing IDs must not be assigned again')
  if not len(ids):return
  # Plan allocation before committing, so validation errors leave state unchanged.
  counters=self.next.copy();free={k:v.copy() for k,v in self.free.items()};physical=np.empty(len(ids),np.uint32)
  # Vectorized per-rank allocation preserves the exact within-rank insertion order.
  for r in range(self.ranks):
   ix=np.flatnonzero(owners==r)
   if not len(ix):continue
   slots=free.setdefault((t,r),[]);take=min(len(ix),len(slots))
   if take:physical[ix[:take]]=np.asarray(slots[-take:][::-1],np.uint32);del slots[-take:]
   count=len(ix)-take;first=int(counters[t,r]);last=(first+count-1)*self.ranks+r
   if count and last>np.iinfo(np.uint32).max:raise ValueError('Physical ID exceeds uint32')
   physical[ix[take:]]=(np.arange(first,first+count,dtype=np.uint64)*self.ranks+r).astype(np.uint32);counters[t,r]+=count
  if self.next_gid+len(ids)>np.iinfo(np.uint32).max:raise ValueError('Logical ID exceeds uint32')
  self._grow(t,int(ids.max())+1);self.gid[t][ids]=np.arange(self.next_gid,self.next_gid+len(ids),dtype=np.uint32);self.next_gid+=len(ids);self.owner[t][ids]=owners;self.physical[t][ids]=physical;self.next=counters;self.free=free;self.storage+=np.bincount(owners,minlength=self.ranks)
 def _allocate(self,t,r,counters,free):
  slots=free.setdefault((t,r),[])
  if slots:return slots.pop()
  value=int(counters[t,r])*self.ranks+r
  if value>np.iinfo(np.uint32).max:raise ValueError('Physical ID exceeds uint32')
  counters[t,r]+=1;return value
 def apply_changes(self,tables,ids,new_owners,*,allowed,max_moved,dim=128,state_tensors=1):
  tables=np.asarray(tables,dtype=np.int64);ids=np.asarray(ids,dtype=np.int64);new=np.asarray(new_owners,dtype=np.int64)
  if tables.shape!=ids.shape or ids.shape!=new.shape or len(set(zip(tables.tolist(),ids.tolist())))!=len(ids):raise ValueError('Non-unique or mismatched migration plan')
  if np.any((tables<0)|(tables>=len(self.owner))) or np.any(ids<0):raise ValueError('Invalid logical ID')
  old=np.array([self.lookup_owner(int(t),[int(fid)])[0] for t,fid in zip(tables,ids)],np.int16)
  if np.any(old<0):raise ValueError('Cannot migrate an unknown ID')
  changed=old!=new;tables=tables[changed];ids=ids[changed];new=new[changed];old=old[changed]
  ledger=migration_ledger(old,new,self.ranks,allowed,max_moved,dim=dim,state_tensors=state_tensors)
  if np.any(np.asarray(ledger['storage_count_delta'])):raise ValueError('Fine-tuning must preserve each rank primary count')
  old_physical=np.array([self.physical[t][fid] for t,fid in zip(tables,ids)],np.uint32);counters=self.next.copy();free={k:v.copy() for k,v in self.free.items()}
  for t,r,slot in zip(tables,old,old_physical):free.setdefault((int(t),int(r)),[]).append(int(slot))
  new_physical=np.array([self._allocate(int(t),int(r),counters,free) for t,r in zip(tables,new)],np.uint32)
  for t,fid,r,slot in zip(tables,ids,new,new_physical):self.owner[t][fid]=r;self.physical[t][fid]=slot
  self.next=counters;self.free=free
  ledger.update(logical_gids=[int(self.gid[t][fid]) for t,fid in zip(tables,ids)],tables=tables.tolist(),original_ids=ids.tolist(),old_owners=old.tolist(),new_owners=new.tolist(),old_physical_ids=old_physical.tolist(),new_physical_ids=new_physical.tolist(),invalidate_cached_logical_ids=True,transport_executed=False)
  return ledger
 def remap(self,t,ids):
  ids=np.asarray(ids,dtype=np.int64)
  if np.any(self.lookup_owner(t,ids)<0):raise ValueError('Unplaced embedding in sample')
  return self.physical[t][ids]
 def audit(self):
  counts=np.zeros(self.ranks,np.int64)
  for t,o in enumerate(self.owner):
   active=np.flatnonzero(o>=0);p=self.physical[t][active];assert len(np.unique(p))==len(p) and np.array_equal(p%self.ranks,o[active]);counts+=np.bincount(o[active],minlength=self.ranks)
   live=set(map(int,p))
   for r in range(self.ranks):
    free=self.free.get((t,r),[]);assert len(set(free))==len(free) and not live.intersection(free);assert all(x%self.ranks==r for x in free)
  assert np.array_equal(counts,self.storage)

 def global_owners(self):
  result=np.full(self.next_gid,-1,np.int16)
  for t,o in enumerate(self.owner):
   ids=np.flatnonzero(o>=0);result[self.gid[t][ids]]=o[ids]
  assert np.all(result>=0);return result
 def profile_gids(self,t,physical_ids):
  active=np.flatnonzero(self.owner[t]>=0);physical=self.physical[t][active];lut=np.full(int(physical.max())+1,np.iinfo(np.uint32).max,np.uint32);lut[physical]=self.gid[t][active];result=lut[np.asarray(physical_ids,dtype=np.int64)];assert np.all(result!=np.iinfo(np.uint32).max);return result
 def save(self,path):
  path=Path(path);path.mkdir(exist_ok=False)
  for t,o in enumerate(self.owner):
   ids=np.flatnonzero(o>=0).astype(np.uint32);np.savez(path/f'table_{t}.npz',original_ids=ids,physical_ids=self.physical[t][ids],owners=o[ids],logical_gids=self.gid[t][ids])
  (path/'manifest.json').write_text(json.dumps(dict(ranks=self.ranks,primary_counts=self.storage.tolist(),next_gid=self.next_gid,next_slots=self.next.tolist(),free_slots={f'{t},{r}':v for (t,r),v in self.free.items() if v}),indent=2))
