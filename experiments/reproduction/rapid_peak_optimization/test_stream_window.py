import tempfile,json
from pathlib import Path
import numpy as np
from stream_window import window
from stream_owner_state import OwnerState
from persistent_replay import *
rng=np.random.default_rng(42)
for g in [8,64]:
 own=OwnerState(g,2)
 for t in range(2):own.add_new(t,np.arange(g*2),np.arange(g*2)%g)
 old=[x.copy() for x in own.owner];n=g*16*3;data=rng.integers(0,g*3,(n,6),dtype=np.uint32)
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'w';cfg=dict(batch_size=16,dense=1,table_lengths=[2,2]);result=window(own,data,cfg,p,rounds=2)
  for t in range(2):assert np.array_equal(old[t],own.owner[t][:len(old[t])])
  trace=trace_from_case(p,own);assignment=np.load(p/'assignment.npy')
  for k,batches in enumerate(trace):
   for r,(ids,c) in enumerate(batches):
    rows=np.flatnonzero(assignment[k*g*16:(k+1)*g*16]==r)+k*g*16
    expected=np.concatenate([own.gid[t][np.unique(data[rows,2+t*2:4+t*2])] for t in range(2)])
    assert np.array_equal(np.sort(ids),np.sort(expected))
  state=State(own.global_owners(),[np.empty(0,np.uint32) for _ in range(g)],0,g);mirrors,saved=select_mirrors(p,state,np.full(g,10),p/'mirrors');set_mirrors(state,mirrors);replay_window(state,trace,expected_saved=saved)
print('WINDOW PIPELINE PASS')
