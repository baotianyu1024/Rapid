import argparse,ctypes,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args();root=Path(__file__).resolve().parent
m=json.load(open(a.result/'result.json'));case=Path(m['case']);cache=Path(m['cache']);g=m['ranks'];nb=m['batches'];hr=np.zeros((5*nb,g),np.int64);hs=hr.copy();mirrors=[np.load(a.result/f'mirrors_{r}.npy') for r in range(g)]
lib=ctypes.CDLL(str(root/'cache_oracle.so'));fn=lib.oracle;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*5;fn.restype=None
ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p);offset=0;start=time.perf_counter()
cfg=json.load(open(case/'config.json'))
for t in range(len(cfg['table_lengths'])):
 owners=np.load(case/f'{t}.owners.npy');n=len(owners);bits=np.zeros(n,np.uint64)
 for r,ids in enumerate(mirrors):
  ix=(ids[(ids>=offset)&(ids<offset+n)]-offset).astype(np.int64);assert np.all(owners[ix]!=r);bits[ix]|=np.uint64(1)<<np.uint64(r)
 ix=np.flatnonzero(bits);prof=np.ascontiguousarray(np.load(cache/f'{t}.profiles.npy',mmap_mode='r')[ix],dtype=np.uint64);own=np.ascontiguousarray(owners[ix],dtype=np.int16);sel=np.ascontiguousarray(bits[ix])
 fn(len(ix),g,nb,*map(ptr,[prof,own,sel,hr,hs]));offset+=n
 print('oracle table',t,'selected IDs',len(ix),'seconds',round(time.perf_counter()-start,1),flush=True)
z=np.load(case/'predicted_counts.npz');r=np.tile(z['receive'].sum(1),(5,1));s=np.tile(z['send'].sum(1),(5,1));actual=np.load(a.result/'counts.npz')
assert np.array_equal(r-hr,actual['fr']) and np.array_equal(s-hs,actual['fs'])
assert np.array_equal(s,actual['br']) and np.array_equal(r,actual['bs'])
(a.result/'verification.json').write_text(json.dumps(dict(success=True,all_selected_ids=True,all_steps_and_ranks=True,epochs=5,independent_algorithm='explicit per-ID master and mirror version state transitions; positive updates represented by one count per active rank (validity depends only on presence), tested against variable occurrence-count State in test_cache.py',seconds=time.perf_counter()-start),indent=2))
print('PASS full real-trace cache oracle')
