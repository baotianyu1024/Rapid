import tempfile,json
from pathlib import Path
import numpy as np
from persistent_replay import *
rng=np.random.default_rng(881)
for g in [8,64]:
 n=150;owner=rng.integers(0,g,n,dtype=np.int16);empty=[np.empty(0,np.uint32) for _ in range(g)];a=State(owner.copy(),empty,0,g);b=State(owner.copy(),empty,0,g)
 for window in range(5):
  changed=np.array([0,1]) if window%2 else np.empty(0,int);own=np.r_[owner,rng.integers(0,g,7,dtype=np.int16)]
  if len(changed):own[changed]=(own[changed]+1)%g
  update_owners(a,own,changed);update_owners(b,own,changed);owner=own;n=len(own)
  mirrors=[np.sort(rng.choice(np.flatnonzero(own!=r),20,replace=False)).astype(np.uint32) for r in range(g)];set_mirrors(a,mirrors);set_mirrors(b,mirrors)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'profile_cache').mkdir();nb=3
   prof=np.zeros((n,g),np.uint64)
   for k in range(nb):
    for r in range(g):prof[rng.choice(n,12,replace=False),r]|=np.uint64(1<<k)
   np.save(p/'profile_cache/0.profiles.npy',prof);np.save(p/'0.logical_gids.npy',np.arange(n,dtype=np.uint32));np.save(p/'0.owners.npy',own)
   (p/'config.json').write_text(json.dumps(dict(ranks=g,batches=nb,table_lengths=[1])))
   selected,expected=select_mirrors(p,a,np.full(g,15),p/'selected');set_mirrors(a,selected);set_mirrors(b,selected)
   trace=[];weighted=[]
   for k in range(nb):
    ids=[np.flatnonzero(prof[:,r]&np.uint64(1<<k)).astype(np.uint32) for r in range(g)]
    trace.append([(x,np.ones(len(x),np.uint16)) for x in ids]);weighted.append([(x,rng.integers(1,20,len(x),dtype=np.uint16)) for x in ids])
   _,aa=replay_window(a,trace,expected_saved=expected);_,bb=replay_window(b,weighted)
   for key in aa:assert np.array_equal(aa[key],bb[key]),key
print('PERSISTENT MULTI-WINDOW CACHE PASS')
