"""Parallel exact sample swaps after owner refinement; fixed batch membership and capacities."""
import argparse,ctypes,json,time,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'scripts'))
from measure import measure
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rounds',type=int,default=2);p.add_argument('--workers',type=int,default=8);p.add_argument('--batches',type=int);a=p.parse_args();a.case=a.case.resolve();a.out=a.out.resolve();a.out.mkdir(exist_ok=False);(a.out/'scheduled').mkdir()
cfg=json.load(open(a.case/'config.json'));g=cfg['ranks'];b=cfg['batch_size'];nb=a.batches or cfg['batches'];assert nb<=cfg['batches'];cfg.update(batches=nb,rows=nb*g*b,base_case=str(a.case));lengths=np.array(cfg['table_lengths'],np.int32);width=1+cfg['dense']+int(lengths.sum())
source=[np.memmap(a.case/f'scheduled/dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width) for r in range(g)]
dest=[np.memmap(a.out/f'scheduled/dataset_{r}.bin',dtype='<u4',mode='w+',shape=(nb*b,width)) for r in range(g)]
lib=ctypes.CDLL(str(ROOT/'sample_peak.so'));fn=lib.sample_peak;fn.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]*2+[ctypes.c_int]+[ctypes.c_void_p]*2;fn.restype=ctypes.c_int;ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
base=np.load(a.case/'cps_counts.npz');bf=base['receive'][:nb].sum(1);bs=base['send'][:nb].sum(1)
start=time.perf_counter()
def step(k):
 records=np.concatenate([x[k*b:(k+1)*b] for x in source]);perm=np.zeros(g*b,np.uint32);fr=np.zeros(g,np.int64);fs=fr.copy()
 moves=fn(ptr(records),g,b,width,cfg['dense'],len(lengths),ptr(lengths),ptr(perm),a.rounds,ptr(fr),ptr(fs))
 assert np.array_equal(np.sort(perm),np.arange(g*b));assert fr.sum()==fs.sum() and fr.sum()<=bf[k].sum();assert fr.max()<=bf[k].max() and fs.max()<=bs[k].max() and (fr+fs).max()<=(bf[k]+bs[k]).max()
 for r in range(g):dest[r][k*b:(k+1)*b]=records[perm[r*b:(r+1)*b]]
 np.save(a.out/f'batch_{k}_permutation.npy',perm)
 print(json.dumps(dict(batch=k,swaps=moves,mean_before=float(bf[k].mean()),mean_after=float(fr.mean()),peak_before=int((bf[k]+bs[k]).max()),peak_after=int((fr+fs).max()),seconds=round(time.perf_counter()-start,1))),flush=True)
 return fr,fs
with ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(step,range(nb)))
for x in dest:x.flush()
metrics,counts=measure([a.out/f'scheduled/dataset_{r}.bin' for r in range(g)],g,b,nb,lengths.tolist(),cfg['dense'],True)
assert np.array_equal(counts['receive'].sum(1),np.array([x[0] for x in results])) and np.array_equal(counts['send'].sum(1),np.array([x[1] for x in results]))
for k in range(nb):
 records=np.concatenate([x[k*b:(k+1)*b] for x in source]);perm=np.load(a.out/f'batch_{k}_permutation.npy')
 for r in range(g):assert np.array_equal(dest[r][k*b:(k+1)*b],records[perm[r*b:(r+1)*b]])
np.savez_compressed(a.out/'cps_counts.npz',**counts);np.savez_compressed(a.out/'predicted_counts.npz',receive=counts['receive'],send=counts['send'])
(a.out/'config.json').write_text(json.dumps(cfg,indent=2));(a.out/'cps_metrics.json').write_text(json.dumps(metrics,indent=2));(a.out/'complete.json').write_text(json.dumps(dict(success=True,all_records_permutation_verified=True,independent_counts_verified=True,rounds=a.rounds,subset=nb!=json.load(open(a.case/'config.json'))['batches'],seconds=time.perf_counter()-start),indent=2))
print('SAMPLE REFINEMENT VERIFIED',flush=True)
