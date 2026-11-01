"""Audited dim=128 comparisons from fresh native plans and complete RAPID."""
from pathlib import Path
import argparse,sys,json
import numpy as np
from evidence_utils import *
from torchrec_count import table_bytes
V=Path(__file__).resolve().parents[1]
def balance(fr,br,full,normalization_steps=None):
 c=fr+br;n=len(c) if normalization_steps is None else normalization_steps
 avg=float(c.sum()/(n*c.shape[1]));peak=float(c.max(axis=1).sum()/n)
 return dict(mean_vectors_per_rank_step=avg,mean_step_peak=peak,peak_over_mean=peak/avg,absolute_peak=int(c.max()),full_steps=int(np.count_nonzero(full)),total_steps=len(c),normalization_steps=n,actual_mean_per_step=float(c.mean()),actual_mean_step_peak=float(c.max(axis=1).mean()))
def rapid(folder,epochs=5):
 cfg=read(folder/'config.json');g=cfg['ranks'];b=cfg['batch_size'];n=cfg['rows'];z=np.load(folder/'cps_counts.npz');m=read(folder/'cps_metrics.json');choose=z['receive'].mean(axis=(0,2))<b*(g-1)
 fr=np.tile(np.where(choose[None,:,None],z['receive'],b*(g-1)).sum(1),(epochs,1));br=np.tile(np.where(choose[None,:,None],z['send'],b*(g-1)).sum(1),(epochs,1));total=int((fr+br).sum());s=read(folder/'storage.json')['primary_counts'];v=dict(total_vectors=total,vector_GiB=total*512/2**30,vectors_per_sample=total/(n*epochs),local_hit=m['unique_local_hit_rate'],reduction_tables=np.flatnonzero(~choose).tolist(),primary_counts=s,primary_max_mean=max(s)/np.mean(s),primary_min=min(s),primary_max=max(s),weight_MiB_max=max(s)/2048,**balance(fr,br,np.ones(len(fr),bool)))
 return v,audit_records(folder)
def report(name):
 case=V/'results'/name;assert read(case/'complete.json')['success'];cfg=read(case/'experiment.json');g=cfg['ranks'];meta=read(case/'data/input.json');b=meta['local_batch'];nb=meta['global_batches'];n=meta['rows'];methods={};audits={};methods['RAPID'],audits['RAPID']=rapid(case/'optimized/peak_samples')
 gm=read(case/'gmp_traffic/het_gmp_s0.json');z=np.load(case/'gmp_traffic/het_gmp_s0.npz');f=z['forward'];back=z['backward'];steps=gm['steps_per_epoch'];owner=np.load(case/'gmp/owner.npy');assign=np.load(case/'gmp/assignment.npy');sp=np.load(case/'data/sparse.npy',mmap_mode='r');rowids=[np.flatnonzero(assign==r) for r in range(g)];ind=np.zeros((steps,g,g),np.int64)
 for k in range(steps):
  for r in range(g):
   ids=np.unique(sp[rowids[r][k*b:(k+1)*b]].ravel());o=owner[ids];ind[k,r]=np.bincount(o[o!=r],minlength=g)
 assert np.array_equal(back,np.tile(ind,(5,1,1))) and np.array_equal(f[0],back[0]) and np.all(f<=back)
 assert gm['pending_updates']==0 and gm['processed_rows']==n*5
 full=np.arange(len(f))%steps<min(gm['rank_rows'])//b;total=int(f.sum()+back.sum());s=gm['primary_counts'];mir=gm['mirror_counts'];methods['HET-GMP']=dict(total_vectors=total,vector_GiB=total*512/2**30,vectors_per_sample=total/(n*5),primary_counts=s,primary_min=min(s),primary_max=max(s),primary_max_mean=max(s)/np.mean(s),weight_MiB_max=max(x+y for x,y in zip(s,mir))/2048,mirror_counts=mir,**balance(f.sum(2),back.sum(1),full,nb*5));audits['HET-GMP']=dict(independent_gradients=True,all_rows_consumed=True,pending_updates=0)
 for key in ['herald','bagpipe']:
  path=case/('bagpipe_la8' if key=='bagpipe' else key)/'result.json'
  if path.exists():
   v=read(path);total=v['total_vectors_with_flush'];methods[{'herald':'Herald','bagpipe':'BagPipe'}[key]]=dict(total_vectors=total,vector_GiB=total*512/2**30,vectors_per_sample=total/(n*5),detail=str(path))
 plan=case/'torchrec/plan.json'
 if plan.exists():
  pl=read(plan);per=[table_bytes(t,g,b,nb*5) for t in pl['tables']];payload=sum(t['forward_bytes']+t['backward_bytes'] for t in per);methods['TorchRec']=dict(total_vectors=payload//512,vector_GiB=payload/2**30,vectors_per_sample=payload/512/(n*5),sharding_counts={k:sum(t['sharding_type']==k for t in pl['tables']) for k in set(t['sharding_type'] for t in pl['tables'])},detail=str(plan));save(case/'torchrec/protocol_bytes.json',per)
 save(case/'comparison.json',dict(config=cfg,methods=methods,verification=audits,scope='FP32 dim128, all remote embedding payload sent once, 5 epochs, includes initial cache filling and final pending gradient writeback; excludes IDs/dense gradients; CPU replay/protocol accounting, not measured GPU traffic',balance_scope='All actual steps including ragged tails retained. Average and accumulated per-step maximum normalized by processed_rows/(ranks*local_batch), i.e. equal-work global-batch equivalents; absolute peak is an actual step.'))
 print(name,{k:round(v['vector_GiB'],6) for k,v in methods.items()},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args()
 if a.case.split('_')[0] in ['t180','t200','t510']:
  assert a.case.endswith('_8'), 'Official T-series reporter currently supports only 8 ranks'
  import subprocess
  subprocess.run([sys.executable,str(V/'official_tseries/code/report.py'),'--dataset',a.case.split('_')[0]],check=True)
 else:report(a.case)
