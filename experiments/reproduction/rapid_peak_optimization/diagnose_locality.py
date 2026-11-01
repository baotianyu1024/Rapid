import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;rows={}
for g in [8,64]:
 case=ROOT/f'r{g}_samples';cache=case/'profile_cache';homecount=np.zeros(g,np.int64);misplaced=np.zeros((g,g),np.int64);storage=np.zeros(g,np.int64)
 for t in range(26):
  ids=np.load(cache/f'{t}.ids.npy',mmap_mode='r');pr=np.load(cache/f'{t}.profiles.npy',mmap_mode='r');owners=ids%g;storage+=np.bincount(owners,minlength=g)
  for start in range(0,len(ids),100000):
   p=pr[start:start+100000];ix=np.flatnonzero(np.count_nonzero(p,axis=1)==1);home=np.argmax(p[ix]!=0,axis=1);actual=owners[start+ix].astype(np.int64);homecount+=np.bincount(home,minlength=g);misplaced+=np.bincount(actual*g+home,minlength=g*g).reshape(g,g)
 rows[g]=dict(exclusive_home_counts=homecount.tolist(),exclusive_owner_to_home=misplaced.tolist(),storage=storage.tolist())
 print(g,'exclusive homes first8',homecount[:8],'returnable first8',(misplaced.sum(0)-np.diag(misplaced))[:8],flush=True)
(ROOT/'locality_diagnosis.json').write_text(json.dumps(rows,indent=2))
