"""Resident-input sequential A/B benchmark and exact owner/traffic differential check."""
import ctypes,json,time,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];cache=root/'balanced_owner_cache32';seed=root/'results/criteo_day6_32_joint_balance_v7';case=root/'results/criteo_day6_32_owner_balanced_16b';out=root/'results/speed_exact';out.mkdir(exist_ok=True)
cfg=json.loads((case/'config.json').read_text());g,nb=cfg['ranks'],cfg['batches'];tables=[]
for t in range(26):
 tables.append(dict(owners=np.load(seed/f'{t}.owners.npy'),profiles=np.asarray(np.load(cache/f'{t}.profiles.npy'),dtype=np.uint64),weights=np.load(cache/f'{t}.weights.npy')))
initial=np.load(seed/'predicted_counts.npz');base=np.load(case/'cps_counts.npz');total0=base['receive'].sum((1,2)).astype(np.int64);norm=total0.astype(float)/g
ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
results=[];reference=None
for version in ['reference','fast','fast','reference']:
 path=root/'scripts'/('joint_price_owners_reference.so' if version=='reference' else 'joint_price_owners.so');fn=ctypes.CDLL(str(path)).refine
 fn.restype=ctypes.c_long;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*10+[ctypes.c_double]*5+[ctypes.c_int]*2
 owners=[t['owners'].copy() for t in tables];rd=initial['receive'].astype(np.int64);sd=initial['send'].astype(np.int64);recv=rd.sum(1);send=sd.sum(1);storage=sum((np.bincount(o,minlength=g) for o in owners),np.zeros(g,dtype=np.int64));seconds=cpu=0.;moved=0
 for t in sorted(range(26),key=lambda t:-sd[:,t,:].sum()):
  r=np.ascontiguousarray(rd[:,t,:]);s=np.ascontiguousarray(sd[:,t,:]);table=tables[t];start=time.perf_counter();startcpu=time.process_time()
  moved+=fn(len(owners[t]),g,nb,*map(ptr,[owners[t],table['profiles'],table['weights'],r,s,recv,send,storage,norm,total0]),8.,32.,60.,.5,.05,0,16)
  cpu+=time.process_time()-startcpu;seconds+=time.perf_counter()-start;rd[:,t,:]=r;sd[:,t,:]=s
 if reference is None:reference=([o.copy() for o in owners],rd.copy(),sd.copy(),storage.copy(),moved)
 else:
  assert all(np.array_equal(x,y) for x,y in zip(reference[0],owners));assert np.array_equal(rd,reference[1]);assert np.array_equal(sd,reference[2]);assert np.array_equal(storage,reference[3]);assert moved==reference[4]
 row=dict(version=version,kernel_wall_seconds=seconds,kernel_cpu_seconds=cpu,moved=moved,all_55874235_owner_assignments_equal=True,all_batch_rank_table_counts_equal=True,library_sha256=hashlib.sha256(path.read_bytes()).hexdigest());results.append(row);print(row,flush=True)
 (out/'owner_kernel_benchmark.json').write_text(json.dumps(dict(results=results,scope='one complete 32-rank pass, inputs resident in RAM, identical v7 initial placement; sequential ABBA; excludes loading/saving/profile/materialization'),indent=2))
