"""Report official generator parts; heterogeneous table dimensions, FP32 payload."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parents[1];V=O.parent;sys.path.insert(0,str(V/'code'))
from evidence_utils import read,save,audit_records
from torchrec_count import table_bytes

def report(ds):
 case=O/f'results/{ds}_8';cfg=read(case/'experiment.json');meta=read(case/'data/input.json');dims=np.array(meta['dimensions']);epochs=5;g=meta['ranks'];b=meta['local_batch'];nb=meta['global_batches'];n=meta['rows'];methods={};audits={}
 def entry(payload,vectors):return dict(total_embedding_bytes=int(payload),vector_GiB=float(payload/2**30),total_vectors=int(vectors),bytes_per_sample=float(payload/(epochs*n)),vectors_per_sample=float(vectors/(epochs*n)))
 folder=case/'optimized/peak_samples';z=np.load(folder/'cps_counts.npz');choose=z['receive'].mean(axis=(0,2))<b*(g-1);recv=np.where(choose[None,:,None],z['receive'],b*(g-1));send=np.where(choose[None,:,None],z['send'],b*(g-1));payload=((recv+send)*dims[None,:,None]*4).sum()*epochs;total=(recv+send).sum()*epochs
 methods['RAPID']=entry(payload,total);methods['RAPID']['reduction_tables']=np.flatnonzero(~choose).tolist();audits['RAPID']=audit_records(folder)
 zz=np.load(case/'gmp_bytes/het_gmp_s0.npz');orig=np.load(case/'gmp_traffic/het_gmp_s0.npz')
 for k in ['forward','backward','queries','pending_flush']:assert np.array_equal(zz[k],orig[k]),k
 assert zz['pending_flush'].sum()==0
 methods['HET-GMP']=entry(zz['forward_bytes'].sum()+zz['backward_bytes'].sum(),zz['forward'].sum()+zz['backward'].sum());audits['HET-GMP']={'byte_instrumentation_preserves_original_vector_counts':True,'terminal_pending':0,'independent_byte_count':read(case/'gmp_bytes/independent_audit.json')};assert audits['HET-GMP']['independent_byte_count']['success']
 for method,directory in [('Herald','herald'),('BagPipe','bagpipe_la8')]:
  val=read(case/directory/'result.json');methods[method]=entry(val['total_embedding_bytes'],val['total_vectors_with_flush']);methods[method]['detail']=str(case/directory/'result.json')
 plan=read(case/'torchrec/plan.json');assert [t['dim'] for t in plan['tables']]==dims.tolist();per=[table_bytes(t,g,b,nb*epochs) for t in plan['tables']];payload=sum(t['forward_bytes']+t['backward_bytes'] for t in per);methods['TorchRec']=dict(total_embedding_bytes=payload,vector_GiB=payload/2**30,bytes_per_sample=payload/(n*epochs),sharding_counts={k:sum(t['sharding_type']==k for t in plan['tables']) for k in set(t['sharding_type'] for t in plan['tables'])});save(case/'torchrec/protocol_bytes.json',per)
 official_methods=methods
 methods={name:dict(values,**entry(values['total_vectors']*512,values['total_vectors'])) for name,values in official_methods.items() if name!='TorchRec'}
 uniform_path=case/'torchrec_uniform128/plan.json'
 while not uniform_path.exists():time.sleep(5)
 uniform_plan=read(uniform_path);assert all(t['dim']==128 for t in uniform_plan['tables']);per=[table_bytes(t,g,b,nb*epochs) for t in uniform_plan['tables']];payload=sum(t['forward_bytes']+t['backward_bytes'] for t in per)
 methods['TorchRec']=dict(**entry(payload,payload//512),sharding_counts={k:sum(t['sharding_type']==k for t in uniform_plan['tables']) for k in set(t['sharding_type'] for t in uniform_plan['tables'])});save(case/'torchrec_uniform128/protocol_bytes.json',per)
 for m in methods.values():assert m['total_embedding_bytes']>0
 result=dict(config=dict(cfg,dim=128,comparison_dimensions='uniform128; official model dimensions reported separately'),methods=methods,official_dimension_methods=official_methods,verification=audits,scope='Official HugeCTR generator: one full 65536-row part, original per-table vocab/hotness and seeded RNG; main table uniform dim128, official per-table model dimensions in separate table; FP32, 8 ranks, local batch512, 5 epochs; active-ID working set for all CPU planners, not full vocabulary training or measured GPU traffic; includes cache fill and modeled terminal writeback, excludes IDs and dense gradients.')
 save(case/'comparison.json',result);print(ds,{k:round(v['vector_GiB'],6) for k,v in methods.items()},flush=True)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['t180','t200','t510','all'],default='all');a=p.parse_args()
 for ds in (['t180','t200','t510'] if a.dataset=='all' else [a.dataset]):report(ds)
