"""Figure4 peer-sparse Reduction, distinct from dense B*G cost in Figure11."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_case import LENGTHS

def peer_counts(ids,ranks,local):
 if ranks>64:raise ValueError('uint64 owner bitset supports <=64 ranks')
 masks=np.bitwise_or.reduce(np.left_shift(np.uint64(1),(ids%ranks).astype(np.uint64)),axis=1)
 masks &= ~np.left_shift(np.uint64(1),np.uint64(local))
 return np.bitwise_count(masks)

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--batches',type=int,default=20);a=p.parse_args()
 data=np.memmap(a.input,dtype='<u4',mode='r').reshape(-1,228);G=8;B=8192
 assert len(data)>=G*B*a.batches
 unique=np.zeros((a.batches,26,G),dtype=np.int64);reduction=np.zeros_like(unique);off=np.cumsum([0]+LENGTHS)
 for b in range(a.batches):
  for r in range(G):
   block=data[(b*G+r)*B:(b*G+r+1)*B,14:]
   for t in range(26):
    ids=block[:,off[t]:off[t+1]];u=np.unique(ids)
    unique[b,t,r]=np.count_nonzero(u%G!=r)
    reduction[b,t,r]=peer_counts(ids,G,r).sum()
  print('Figure4 batch',b+1,'/',a.batches,flush=True)
 u=unique.mean(axis=(0,2));red=reduction.mean(axis=(0,2));acp=np.minimum(u,red)
 result=dict(input=str(Path(a.input).resolve()),ranks=G,batch_size=B,batches=a.batches,rows=G*B*a.batches,
   reduction_model='one vector per sample and nonempty remote owner; original analysis/compute_unique_vs_reduction.py',
   per_table_unique=u.tolist(),per_table_reduction=red.tolist(),per_table_acp=acp.tolist(),
   total_unique=float(u.sum()),total_reduction=float(red.sum()),total_acp=float(acp.sum()),
   table20_unique_over_reduction=float(u[20]/red[20]),acp_over_unique=float(acp.sum()/u.sum()))
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));np.savez_compressed(out.with_suffix('.npz'),unique=unique,reduction=reduction)
if __name__=='__main__':main()
