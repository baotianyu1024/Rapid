"""Exact communication-delta sample swap experiment with fixed owners and batch capacities."""
import argparse,ctypes,json,time
from pathlib import Path
import numpy as np
from measure import measure,summarize
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--batches',type=int);p.add_argument('--rounds',type=int,default=4);a=p.parse_args()
case=Path(a.case).resolve();out=Path(a.output).resolve();out.mkdir(exist_ok=False);(out/'scheduled').mkdir()
cfg=json.loads((case/'config.json').read_text());g,b=cfg['ranks'],cfg['batch_size'];nb=min(a.batches or cfg['batches'],cfg['batches']);lengths=np.array(cfg['table_lengths'],dtype=np.int32);width=1+cfg['dense']+sum(lengths)
arrays=[np.memmap(case/'scheduled'/f'dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width) for r in range(g)]
base=np.load(case/'cps_counts.npz');bf=base['receive'][:nb].sum(1);bb=base['send'][:nb].sum(1);fr=float((bf.max(1)/bf.mean(1)).max());br=float((bb.max(1)/bb.mean(1)).max())
lib=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'speed_work/fast/sample_refine.so'));fn=lib.sample_refine;fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]*2+[ctypes.c_int]+[ctypes.c_double]*2+[ctypes.c_void_p]*2
ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p);files=[(out/'scheduled'/f'dataset_{r}.bin').open('wb') for r in range(g)];predf=[];predb=[];seconds=0;history=[]
try:
 for step in range(nb):
  records=np.concatenate([x[step*b:(step+1)*b] for x in arrays]);assignment=np.empty(g*b,dtype=np.uint32);f=np.zeros(g,dtype=np.int64);back=f.copy();t=time.perf_counter()
  moves=fn(ptr(records),g,b,width,cfg['dense'],len(lengths),ptr(lengths),ptr(assignment),a.rounds,fr,br,ptr(f),ptr(back));seconds+=time.perf_counter()-t
  assert np.array_equal(np.sort(assignment),np.arange(g*b));np.save(out/f'batch_{step}_permutation.npy',assignment)
  for r in range(g):records[assignment[r*b:(r+1)*b]].tofile(files[r])
  predf.append(f);predb.append(back);row=dict(batch=step,swaps=moves,total_before=int(bf[step].sum()),total_after=int(f.sum()));history.append(row);print(row,flush=True)
finally:
 for f in files:f.close()
metrics,actual=measure([out/'scheduled'/f'dataset_{r}.bin' for r in range(g)],g,b,nb,lengths.tolist(),cfg['dense'],True)
assert np.array_equal(actual['receive'].sum(1),predf) and np.array_equal(actual['send'].sum(1),predb)
for step in range(nb):
 records=np.concatenate([x[step*b:(step+1)*b] for x in arrays]);permutation=np.load(out/f'batch_{step}_permutation.npy')
 for r in range(g):
  x=np.memmap(out/'scheduled'/f'dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width)
  assert np.array_equal(x[step*b:(step+1)*b],records[permutation[r*b:(r+1)*b]])
np.savez_compressed(out/'cps_counts.npz',**actual)
(out/'cps_metrics.json').write_text(json.dumps(metrics,indent=2));(out/'config.json').write_text(json.dumps(dict(cfg,batches=nb,rows=nb*g*b,base_case=str(case)),indent=2))
report=dict(base_case=str(case),batches=nb,subset_screen=nb<cfg['batches'],kernel_seconds=seconds,history=history,before_forward=summarize(bf),before_backward=summarize(bb),after_forward=summarize(np.array(predf)),after_backward=summarize(np.array(predb)),traffic_change=float(np.sum(predf)/bf.sum()-1),all_records_permutation_verified=True,independent_counts_verified=True)
(out/'sample_refinement.json').write_text(json.dumps(report,indent=2));print('DONE',report['traffic_change'],flush=True)
