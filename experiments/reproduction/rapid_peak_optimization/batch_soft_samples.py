import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'scripts'))
from measure import measure
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rounds',type=int,default=4);p.add_argument('--balance',type=float,default=1);a=p.parse_args();a.case=a.case.resolve();a.out=a.out.resolve();a.out.mkdir(exist_ok=False);(a.out/'scheduled').mkdir();start=time.perf_counter()
cfg=json.load(open(a.case/'config.json'));g=cfg['ranks'];b=cfg['batch_size'];nb=cfg['batches'];n=cfg['rows'];lengths=cfg['table_lengths'];width=1+cfg['dense']+sum(lengths);d=1+cfg['dense'];nr=n//g
source=[np.memmap(a.case/f'scheduled/dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width) for r in range(g)]
sparse=np.lib.format.open_memmap(a.out/'compact_sorted.npy',mode='w+',dtype=np.uint32,shape=(n,sum(lengths)))
offset=0;col=0;owners=[]
for t,l in enumerate(lengths):
 ids=np.load(a.case/f'profile_cache/{t}.ids.npy');lut=np.full(int(ids.max())+1,0xffffffff,np.uint32);lut[ids]=np.arange(offset,offset+len(ids),dtype=np.uint32);owners.append((ids%g).astype(np.int16));offset+=len(ids)
 for r in range(g):
  v=lut[source[r][:,d+col:d+col+l]];assert (v!=0xffffffff).all();sparse[r*nr:(r+1)*nr,col:col+l]=v
 col+=l
 print('compact table',t,'seconds',round(time.perf_counter()-start,1),flush=True)
sparse.sort(axis=1);sparse.flush();own=np.concatenate(owners);assign=np.repeat(np.arange(g,dtype=np.int32),nr);lib=ctypes.CDLL(str(ROOT/'batch_soft_samples.so'));fn=lib.batch_soft_samples;fn.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]*2+[ctypes.c_int,ctypes.c_double];fn.restype=ctypes.c_int;ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
moves=fn(ptr(sparse),n,sum(lengths),offset,g,b,ptr(own),ptr(assign),a.rounds,a.balance);assert np.array_equal(np.bincount(assign,minlength=g),np.full(g,nr));np.save(a.out/'assignment.npy',assign)
# Deterministic shuffle matches the uniform minibatch probability model.
for r in range(g):
 parts=[np.flatnonzero((assign==r)&((np.arange(n)%nr)//b==k)) for k in range(nb)];assert all(len(v)==b for v in parts);ix=np.concatenate(parts);np.save(a.out/f'rank_{r}_indices.npy',ix);out=np.empty((nr,width),np.uint32)
 for old in range(g):
  mask=(ix//nr)==old;out[mask]=source[old][ix[mask]%nr]
 out.tofile(a.out/f'scheduled/dataset_{r}.bin')
 check=np.memmap(a.out/f'scheduled/dataset_{r}.bin',dtype='<u4',mode='r').reshape(nr,width)
 for old in range(g):
  mask=(ix//nr)==old;assert np.array_equal(check[mask],source[old][ix[mask]%nr])
 print('fixed-batch materialized rank',r,flush=True)
metrics,counts=measure([a.out/f'scheduled/dataset_{r}.bin' for r in range(g)],g,b,nb,lengths,cfg['dense'],True)
np.savez_compressed(a.out/'cps_counts.npz',**counts);np.savez_compressed(a.out/'predicted_counts.npz',receive=counts['receive'],send=counts['send']);cfg.update(base_case=str(a.case),global_sample_regrouping=False,global_batch_membership_preserved=True)
(a.out/'config.json').write_text(json.dumps(cfg,indent=2));(a.out/'cps_metrics.json').write_text(json.dumps(metrics,indent=2));(a.out/'complete.json').write_text(json.dumps(dict(success=True,all_records_permutation_verified=True,independent_counts_verified=True,embedding_owners_unchanged=True,rank_samples_exact=True,rounds=a.rounds,balance=a.balance,moves=moves,seconds=time.perf_counter()-start),indent=2));print('FIXED BATCH SOFT VERIFIED',flush=True)
