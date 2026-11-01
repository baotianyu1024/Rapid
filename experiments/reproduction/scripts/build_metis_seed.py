"""Paper-style small co-occurrence graph initialization, independent of plot targets."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import pymetis
from run_case import LENGTHS
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--hot',required=True);p.add_argument('--output',required=True);p.add_argument('--rows',type=int,default=2048);p.add_argument('--sample-scope-rows',type=int,default=0,help='Uniformly sample this prefix instead of taking its first rows');p.add_argument('--sample-seed',type=int,default=42);p.add_argument('--ranks',type=int,nargs='+',default=[8,32,128]);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
all_rows=Path(a.input).stat().st_size//912
mm=np.memmap(a.input,dtype=np.int32,mode='r',shape=(all_rows,228))
if a.sample_scope_rows:
 scope=min(a.sample_scope_rows,all_rows)
 if a.rows>scope: raise ValueError('--rows cannot exceed --sample-scope-rows')
 row_indices=np.sort(np.random.default_rng(a.sample_seed).choice(scope,size=a.rows,replace=False))
 x=np.asarray(mm[row_indices,14:])
else:
 row_indices=np.arange(min(a.rows,all_rows))
 x=mm[:a.rows,14:]
encoded=[];offset=0;hot_hash={}
for t,k in enumerate(LENGTHS):
 f=Path(a.hot)/f'{t}.txt';hot_hash[str(t)]=hashlib.sha256(f.read_bytes()).hexdigest()
 hot=np.array([int(s.split(':')[0]) for s in f.read_text().splitlines()],dtype=np.int64)
 v=x[:,offset:offset+k].astype(np.int64);offset+=k
 v[np.isin(v,hot)]=-1
 encoded.append(np.where(v>=0,v+(t<<32),-1))
x=np.concatenate(encoded,axis=1)
nodes=np.unique(x[x>=0]);print('nodes',len(nodes),flush=True)
edges=[]
for row in x:
 ids=np.searchsorted(nodes,np.unique(row[row>=0]));i,j=np.triu_indices(len(ids),1)
 edges.append(ids[i].astype(np.uint64)*len(nodes)+ids[j])
e,w=np.unique(np.concatenate(edges),return_counts=True)
u=(e//len(nodes)).astype(np.int64);v=(e%len(nodes)).astype(np.int64)
src=np.concatenate([u,v]);dst=np.concatenate([v,u]);weights=np.concatenate([w,w]);order=np.argsort(src,kind='stable')
starts=np.r_[0,np.cumsum(np.bincount(src,minlength=len(nodes)))].astype(pymetis.zero_copy_dtype());adj=dst[order].astype(pymetis.zero_copy_dtype());weights=weights[order].astype(pymetis.zero_copy_dtype())
np.savez_compressed(out/'graph.npz',nodes=nodes,starts=starts,adjacency=adj,weights=weights)
meta=dict(input=str(Path(a.input).resolve()),rows=a.rows,sample_scope_rows=a.sample_scope_rows or a.rows,sample_seed=a.sample_seed,row_indices_sha256=hashlib.sha256(row_indices.astype(np.int64).tobytes()).hexdigest(),hot_sha256=hot_hash,nodes=len(nodes),undirected_edges=len(e),seed=42,method='undirected cold-ID co-occurrence clique per sample, integer occurrence weights; uniformly sampled without replacement when sample_scope_rows is set',pymetis='2025.2.2',partitions={})
for g in a.ranks:
 start=time.time();part=pymetis.part_graph(g,adjacency=pymetis.CSRAdjacency(starts,adj),eweights=weights,options=pymetis.Options(seed=42))
 owners=np.asarray(part.vertex_part);np.savetxt(out/f'seed_{g}.txt',np.c_[nodes>>32,nodes&0xffffffff,owners],fmt='%d')
 meta['partitions'][g]=dict(edge_cut=part.edge_cuts,counts=np.bincount(owners,minlength=g).tolist(),seconds=time.time()-start)
 (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(g,meta['partitions'][g],flush=True)
