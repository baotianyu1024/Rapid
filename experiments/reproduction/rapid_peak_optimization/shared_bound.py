import sys,json,ctypes
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;case=Path(sys.argv[1]);cfg=json.load(open(case/'config.json'));g=cfg['ranks'];nb=cfg['batches'];result=np.zeros((nb,g),np.int64);lib=ctypes.CDLL(str(root/'shared_bound.so'));fn=lib.shared_bound;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
for t in range(26):
 p=np.asarray(np.load(case/f'profile_cache/{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64);w=np.load(case/f'profile_cache/{t}.weights.npy',mmap_mode='r');fn(len(p),g,nb,*map(ptr,[p,w,result]))
(case/'uncached_shared_bound.json').write_text(json.dumps(dict(note='Fixed sample schedule, no-cache lower bound on per-rank forward-receive plus backward-receive: each ID queried by two or more ranks costs at least one vector at every querying rank, independent of owner choice. Does not bound cached communication.',rank_means=result.mean(0).tolist(),mean_step_peak=float(result.max(1).mean()),counts=result.tolist()),indent=2));print('uncached bound',result.max(1).mean(),sorted(enumerate(result.mean(0)),key=lambda x:-x[1])[:5])
