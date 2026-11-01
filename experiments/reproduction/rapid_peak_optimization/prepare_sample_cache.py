import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'scripts'))
from refine_balanced_owners import build_cache
p=argparse.ArgumentParser();p.add_argument('case',type=Path);a=p.parse_args();case=a.case.resolve();cfg=json.load(open(case/'config.json'));g=cfg['ranks'];cache=case/'profile_cache';build_cache(case,cache,cfg);storage=np.zeros(g,np.int64)
for t in range(len(cfg['table_lengths'])):
 ids=np.load(cache/f'{t}.ids.npy');owners=(ids%g).astype(np.int16);np.save(case/f'{t}.owners.npy',owners);storage+=np.bincount(owners,minlength=g)
(case/'storage.json').write_text(json.dumps(dict(primary_counts=storage.tolist(),max_over_mean=float(storage.max()/storage.mean()),min_over_mean=float(storage.min()/storage.mean())),indent=2))
print('CACHE READY',flush=True)
