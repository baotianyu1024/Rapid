"""Compose every verified remap back to original feature IDs for streaming use."""
import argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.case=a.case.resolve();a.out.mkdir(exist_ok=False);chain=[];cur=a.case
while True:
 cfg=json.load(open(cur/'config.json'));chain.append(cur)
 if not cfg.get('base_case'):break
 cur=Path(cfg['base_case'])
g=cfg['ranks'];root=Path(__file__).resolve().parent;data=root.parent/'het_gmp_rapid/results/main64/data';ids=[np.load(data/f'ids_{t}.npy') for t in range(26)];original_owner=[np.full(int(v.max())+1,-1,np.int16) for v in ids];start=time.perf_counter()
for r in range(g):
 values=np.loadtxt(cur/f'placement/table_{r}',delimiter=':',dtype=np.uint32,ndmin=2)
 for t in range(26):
  f=values[values[:,0]==t,1];assert np.all(original_owner[t][f]<0);original_owner[t][f]=r
 print('placement rank',r,flush=True)
counts=np.zeros(g,np.int64)
for t,v in enumerate(ids):
 own=original_owner[t][v];assert np.all(own>=0);physical=np.empty(len(v),np.uint32)
 for r in range(g):
  ix=np.flatnonzero(own==r);physical[ix]=np.arange(len(ix),dtype=np.uint32)*g+r
 for case in reversed(chain[:-1]):
  file=case/f'table_{t}_remap.npz'
  if file.exists():
   z=np.load(file);lut=np.full(int(z['old'].max())+1,0xffffffff,np.uint32);lut[z['old']]=z['new'];physical=lut[physical];assert np.all(physical!=0xffffffff)
 assert len(np.unique(physical))==len(physical);own=(physical%g).astype(np.int16);counts+=np.bincount(own,minlength=g)
 np.savez(a.out/f'table_{t}.npz',original_ids=v,physical_ids=physical,owners=own)
 print('exported table',t,flush=True)
if (a.case/'storage.json').exists():assert np.array_equal(counts,json.load(open(a.case/'storage.json'))['primary_counts'])
(a.out/'manifest.json').write_text(json.dumps(dict(case=str(a.case),chain=list(map(str,chain)),ranks=g,primary_counts=counts.tolist(),original_ids_preserved=True,mapping_bijective=True,seconds=time.perf_counter()-start),indent=2))
