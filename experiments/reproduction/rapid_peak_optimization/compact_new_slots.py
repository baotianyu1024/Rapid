"""Compact only not-yet-trained additions into free slots; old addresses stay fixed."""
import numpy as np

def compact_new_slots(state,old_n):
 for t in range(len(state.owner)):
  ids=np.flatnonzero((state.owner[t]>=0)&(state.gid[t]>=old_n));owners=state.owner[t][ids]
  for r in range(state.ranks):
   selected=ids[owners==r];available=np.r_[state.physical[t][selected],np.array(state.free.get((t,r),[]),np.uint32)];available.sort();assert np.all(np.diff(available.astype(np.int64))>0)
   state.physical[t][selected]=available[:len(selected)];free=available[len(selected):];keep=len(free);counter=int(state.next[t,r])
   while keep and int(free[keep-1])==(counter-1)*state.ranks+r:keep-=1;counter-=1
   state.next[t,r]=counter;state.free[(t,r)]=free[:keep].tolist()

def storage_accounting(state):
 free=np.zeros(state.ranks,np.int64)
 for (t,r),slots in state.free.items():free[r]+=len(slots)
 allocated=state.next.sum(0);assert np.array_equal(allocated-free,state.storage)
 return dict(primary_rows=state.storage.tolist(),allocated_physical_slots=allocated.tolist(),free_physical_slots=free.tolist(),allocated_slot_max_over_mean=float(allocated.max()/allocated.mean()))
