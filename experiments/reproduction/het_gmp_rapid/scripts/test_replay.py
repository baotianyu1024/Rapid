"""Hand-computed independent checks of cache validity, deferred updates, and bytes."""
import numpy as np
from replay_het import State

owner=np.array([0],dtype=np.int32)
mirrors=[np.array([],dtype=np.int32),np.array([0],dtype=np.int32)]
batches=[(np.array([0]),np.array([1])),(np.array([0]),np.array([2]))]
s=State(owner,mirrors,3,2)
for _ in range(5):s.step(batches)
r,raw=s.finish()
assert raw['forward'][:,1,0].tolist()==[1,0,0,0,1]
assert raw['backward'][:,1,0].tolist()==[0,1,0,1,0]
assert r['pending_updates']==2 and r['pending_embedding_vectors']==1
assert r['metadata_receive_bytes']['total']==408
assert r['updates']==15 and s.master.tolist()==[14]
for cache in [mirrors,[np.array([],dtype=np.int32)]*2]:
    s=State(owner,cache,0,2)
    for _ in range(5):s.step(batches)
    r,_=s.finish()
    assert r['forward_receive_vectors']['total']==5
    assert r['backward_receive_vectors']['total']==5
    assert r['pending_updates']==0
# Different embeddings must not alias; local requests generate no vector traffic.
s=State(np.array([0,1]),[np.array([],dtype=np.int32)]*2,0,2)
s.step([(np.array([0]),np.array([7])),(np.array([1]),np.array([4]))])
r,_=s.finish()
assert r['forward_receive_vectors']['total']==r['backward_receive_vectors']['total']==0
assert r['updates']==11
print('PASS: cold miss, equality threshold, refresh, duplicate counts, accumulated push, final pending, metadata, local exclusion')
