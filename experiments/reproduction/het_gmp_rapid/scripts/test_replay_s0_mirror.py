"""Hand-derived s=0 mirror traffic: cache validity is not blanket local-hit exemption."""
import numpy as np
from replay_het import State
empty=(np.array([],dtype=np.int32),np.array([],dtype=np.int64))
remote=(np.array([0],dtype=np.int32),np.array([3],dtype=np.int64))
local=(np.array([0],dtype=np.int32),np.array([2],dtype=np.int64))
s=State(np.array([0],dtype=np.int32),[np.array([],dtype=np.int32),np.array([0],dtype=np.int32)],0,2)
# First miss; sole-writer mirror remains current next round; an owner update in
# round3 makes it stale for round4. Duplicate uses still send just one vector.
for owner_batch in [empty,empty,local,empty]:s.step([owner_batch,remote])
meta,z=s.finish()
assert z['forward'][:,1,0].tolist()==[1,0,0,1]
assert z['backward'][:,1,0].tolist()==[1,1,1,1]
assert meta['pending_updates']==0 and meta['updates']==14
print('PASS s=0: sole-writer mirror hit, other-writer invalidation, duplicate gradient aggregation, no pending gradients')
