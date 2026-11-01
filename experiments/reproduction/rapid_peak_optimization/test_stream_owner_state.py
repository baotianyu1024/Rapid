import numpy as np,json,tempfile
from pathlib import Path
from stream_owner_state import OwnerState
for g in [8,64]:
 s=OwnerState(g,2);s.add_new(0,np.arange(g*3),np.arange(g*3)%g);s.add_new(1,np.arange(g*2),np.arange(g*2)%g);s.audit()
 old=[x.copy() for x in s.owner];oldphysical=[x.copy() for x in s.physical]
 s.add_new(0,np.arange(g*3,g*4),np.arange(g)%g)
 for t in range(2):assert np.array_equal(s.owner[t][:len(old[t])],old[t]) and np.array_equal(s.physical[t][:len(oldphysical[t])],oldphysical[t])
 for allowed,budget in [(False,2),(True,1)]:
  before=[x.copy() for x in s.owner]
  try:s.apply_changes([0,1],[0,1],[1,0],allowed=allowed,max_moved=budget)
  except ValueError:pass
  else:raise AssertionError('Missing policy rejection')
  assert all(np.array_equal(a,b) for a,b in zip(before,s.owner))
 counts=s.storage.copy();x=s.apply_changes([0,1],[0,1],[1,0],allowed=True,max_moved=2,state_tensors=3);s.audit();assert x['total_network_tensor_bytes']==3072 and np.array_equal(counts,s.storage)
 x=s.apply_changes([0,0],[2,3],[3,2],allowed=True,max_moved=2);s.audit();assert x['new_physical_ids']==x['old_physical_ids'][::-1]
 s.add_new(1,[g*2],[1]);s.audit()
 with tempfile.TemporaryDirectory() as temp:
  s.save(Path(temp)/'state');loaded=OwnerState.from_export(Path(temp)/'state');loaded.audit();assert np.array_equal(s.global_owners(),loaded.global_owners());assert {k:v for k,v in s.free.items() if v}==loaded.free
  for t in range(2):assert np.array_equal(s.gid[t],loaded.gid[t])
Path(__file__).with_name('stream_owner_state_test.json').write_text(json.dumps(dict(success=True,ranks=[8,64],checks=['old logical owners and physical IDs frozen on new-ID insertion','atomic rejection of unauthorized/over-budget migrations','same-table and cross-table slot reuse without collisions','per-rank primary counts invariant under fine-tuning','both moved embeddings charged; Adam state included']),indent=2));print('STREAM OWNER STATE PASS')
