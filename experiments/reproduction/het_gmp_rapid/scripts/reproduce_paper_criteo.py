"""Paper-aligned Criteo 1D ablation: native estimator vs exact full-epoch ID trace."""
import ctypes,hashlib,json,time,sys
from pathlib import Path
import numpy as np
import hetuCTR_partition
from common import save
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'reproduction/het_gmp_rapid/results/paper_criteo8';P=8;B=8192

def main():
 meta=json.loads((OUT/'vocabulary.json').read_text());n=meta['rows'];train=n-n//10
 assert n==45840617 and meta['embeddings']==33762577,(n,meta['embeddings'])
 start=time.perf_counter();raw=np.memmap(OUT/'categories_local.bin',dtype=np.int32,mode='r',shape=(n,26))
 rng=np.random.RandomState(42);perm=rng.permutation(n);np.save(OUT/'train_source_rows.npy',perm[:train]);luts=[np.fromfile(OUT/f'lut_{t}.bin',dtype=np.int32) for t in range(26)]
 data=np.lib.format.open_memmap(OUT/'train_sparse.npy',dtype=np.int32,mode='w+',shape=(train,26))
 for off in range(0,train,131072):
  block=raw[perm[off:min(off+131072,train)]]
  for t in range(26):block[:,t]=luts[t][block[:,t]]
  data[off:off+len(block)]=block
  if off%1048576==0:print('shuffle and map',off,'/',train,flush=True)
 data.flush();del perm,luts,raw
 save(OUT/'config.json',dict(rows_full=n,rows_train=train,fields=26,active_vocabulary_full=meta['embeddings'],ranks=P,batch_size=B,shuffle_seed=42,split='90% train / 10% validation, matching upstream loader; upstream shuffle seed unspecified',encoding='raw strings, missing -1, per-table sorted LabelEncoder-equivalent IDs; no modulo, no multihot',scope='Table3-style random vs 1D rounds1/3/5, no mirrors; not Figure8 s100 or physical NCCL',input_preparation_seconds=time.perf_counter()-start))
 counts=np.bincount(data.ravel());assert counts.max()<train;del counts
 ctypes.CDLL(None).srand(1);mat=np.full((P,P),.1,dtype=np.float32);np.fill_diagonal(mat,0)
 start=time.perf_counter();part=hetuCTR_partition.partition(data,mat,P,B);construction=time.perf_counter()-start
 records=[]
 def snapshot(i,td=0,te=0):
  cost=np.asarray(part.get_communication()).copy();np.fill_diagonal(cost,0)
  assign,owner=part.get_result();np.save(OUT/f'assignment_{i}.npy',assign);np.save(OUT/f'owner_{i}.npy',owner)
  rec=dict(round=i,native_estimated_vectors_per_global_step=float(cost.sum(dtype=np.float64)),data_seconds=td,embedding_seconds=te,data_counts=np.bincount(assign,minlength=P).tolist(),embedding_counts=np.bincount(owner,minlength=P).tolist())
  records.append(rec);save(OUT/'native_partition.json',dict(graph_seconds=construction,rounds=records));print('SNAPSHOT',json.dumps(rec),flush=True)
 snapshot(0)
 for i in range(1,6):
  t=time.perf_counter();part.refine_data();td=time.perf_counter()-t;t=time.perf_counter();part.refine_embed();te=time.perf_counter()-t
  print('partition round',i,'data',td,'embedding',te,flush=True)
  if i in [1,3,5]:snapshot(i,td,te)
 del part
 # Independently enumerate every real batch: one remotely sent unique vector per
 # requesting rank and ID. With no mirrors and bound0, backward is the transpose.
 for i in [0,1,3,5]:
  t=time.perf_counter();assign=np.load(OUT/f'assignment_{i}.npy');owner=np.load(OUT/f'owner_{i}.npy');rows=[np.flatnonzero(assign==r) for r in range(P)]
  steps=max((len(x)+B-1)//B for x in rows);full=min(len(x)//B for x in rows)
  trace=np.zeros((steps,P,P),dtype=np.int64);expected=np.zeros((P,P));E=len(owner)
  for r,ix in enumerate(rows):
   for b in range((len(ix)+B-1)//B):
    ids=np.unique(data[ix[b*B:(b+1)*B]].ravel());trace[b,r]=np.bincount(owner[ids],minlength=P);trace[b,r,r]=0
   # Independent float64 frequency model; use actual rank sample counts.
   freq=np.bincount(data[ix].ravel(),minlength=E);nonzero=np.flatnonzero(freq)
   q=np.minimum(freq[nonzero]/len(ix),1.)
   with np.errstate(divide='ignore'):prob=-np.expm1(B*np.log1p(-q))
   expected[r]=np.bincount(owner[nonzero],weights=prob,minlength=P);expected[r,r]=0
   print('trace round',i,'rank',r,'verified',flush=True)
  assert len(assign)==train and sum(map(len,rows))==train and np.all(np.diagonal(trace,axis1=1,axis2=2)==0)
  f=trace.sum(2);g=trace.sum(1);assert np.array_equal(f.sum(1),g.sum(1))
  result=dict(round=i,processed_rows=train,steps=steps,full_steps=full,forward_epoch_vectors=int(trace.sum()),forward_full_step_mean=float(trace[:full].sum((1,2)).mean()),forward_rank_step_mean=float(f[:full].mean()),forward_peak_mean=float(f[:full].max(1).mean()),backward_peak_mean=float(g[:full].max(1).mean()),float64_binomial_expected_vectors_per_step=float(expected.sum()),seconds=time.perf_counter()-t)
  save(OUT/f'trace_{i}.json',result);np.savez_compressed(OUT/f'trace_{i}.npz',forward_matrix=trace,forward_receive=f,backward_receive=g)
  print('TRACE',json.dumps(result),flush=True)
 save(OUT/'complete.json',dict(success=True));print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
