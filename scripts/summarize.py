"""Report remote vector payload; never double-count all-rank sends plus receives."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dim',type=int,default=128);p.add_argument('--epochs',type=int,default=5);a=p.parse_args()
 cfg=json.loads((a.case/'config.json').read_text());c=np.load(a.case/'cps_counts.npz');r=c['receive'];s=c['send'];assert np.array_equal(r.sum((1,2)),s.sum((1,2)))
 cap=cfg['batch_size']*(cfg['ranks']-1);choose=r.mean((0,2))<cap
 fr=np.where(choose[None,:,None],r,cap).sum(1);fs=np.where(choose[None,:,None],s,cap).sum(1)
 # Forward receive + backward receive = forward receive + forward send.
 endpoint=fr+fs;scale=a.dim*4
 result=dict(metric='FP32 embedding/gradient payload; IDs, routing, optimizer and dense-model traffic excluded',dim=a.dim,epochs=a.epochs,gpu_execution=False,all_rank_forward_backward_GiB=float(2*fr.sum()*scale*a.epochs/2**30),mean_rank_forward_backward_vectors_per_step=float(endpoint.mean()),mean_step_peak_rank_vectors=float(endpoint.max(1).mean()),absolute_peak_rank_vectors=int(endpoint.max()),peak_of_rank_means_vectors=float(endpoint.mean(0).max()),peak_mean_ratio=float(endpoint.max(1).mean()/endpoint.mean()) if endpoint.mean() else 1.,rank_mean_vectors=endpoint.mean(0).tolist(),acp_unique_tables=np.flatnonzero(choose).tolist())
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
