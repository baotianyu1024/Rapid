"""Call unmodified upstream partitioner; save ownership and sorted mirror lists."""
import argparse, ctypes, time
import numpy as np
import hetuCTR_partition
from common import *

p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--rounds',type=int,default=10)
p.add_argument('--ranks',type=int,default=P); p.add_argument('--batch-size',type=int,default=B)
a=p.parse_args(); P=a.ranks; B=a.batch_size
assert P > 0 and B > 0
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
ctypes.CDLL(None).srand(1)
data=np.load(Path(a.data)/'sparse.npy',mmap_mode='r')
guard_start=time.perf_counter()
frequency=np.bincount(data.ravel())
assert frequency.max() < np.iinfo(np.int32).max  # lookup allocation bounds fixed, algorithm unchanged
maximum_degree=int(frequency.max()); del frequency
guard_seconds=time.perf_counter()-guard_start
mat=np.full((P,P),0.1,dtype=np.float32); np.fill_diagonal(mat,0)
t=time.perf_counter(); part=hetuCTR_partition.partition(data,mat,P,B)
construction=time.perf_counter()-t; history=[]
print('graph constructed',construction,flush=True)
print('initial cost',float(np.sum(np.asarray(part.get_communication())*mat)),flush=True)
for i in range(a.rounds):
    t=time.perf_counter(); part.refine_data(); td=time.perf_counter()-t
    t=time.perf_counter(); part.refine_embed(); te=time.perf_counter()-t
    diag_start=time.perf_counter();cost=np.asarray(part.get_communication())
    v=dict(round=i+1,data_seconds=td,embedding_seconds=te,diagnostic_seconds=time.perf_counter()-diag_start,weighted_cost=float(np.sum(cost*mat)),data_counts=np.asarray(part.get_data_cnt()).tolist(),embedding_counts=np.asarray(part.get_embed_cnt()).tolist())
    history.append(v); print(v,flush=True)
t=time.perf_counter()
assignment,owner=part.get_result()
assert len(assignment)==len(data) and np.all((assignment>=0)&(assignment<P))
np.save(out/'assignment.npy',assignment); np.save(out/'owner.npy',owner)
priority=part.get_priority()
mirror_counts=[]
for r in range(P):
    idxs=np.where(owner==r)[0]
    priority[r][idxs]=-1
    # Match original partition.py's full argsort and training --store_rate=0.01.
    ranked=np.argsort(priority[r])[len(idxs):][::-1]
    mirrors=np.sort(ranked[:int(.01*len(ranked))]).astype(np.int32)
    assert np.all(owner[mirrors]!=r)
    np.save(out/f'mirrors_{r}.npy',mirrors); mirror_counts.append(len(mirrors))
save(out/'partition.json',dict(ranks=P,local_batch=B,communication_matrix=mat.tolist(),graph_seconds=construction,rounds=history,export_priority_seconds=time.perf_counter()-t,guard_seconds=guard_seconds,maximum_degree=maximum_degree,store_rate=.01,mirror_counts=mirror_counts,seed=1,original_omp_threads=16,note='Upstream OpenMP shared-state heuristic can be nondeterministic; saved output defines replay.'))
