import sys,json
from pathlib import Path
import numpy as np
from cache_extension import masks
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'het_gmp_rapid/scripts'))
from replay_het import State
rng=np.random.default_rng(20261026)
for g,nb in [(8,8),(8,64),(64,8)]:
 for trial in range(8):
  n=1000;counts=rng.integers(1,9,size=(nb,g,n))*(rng.random((nb,g,n))<.08);owner=rng.integers(g,size=n)
  prof=np.zeros((n,g),dtype=np.uint64)
  for b in range(nb):prof|=(counts[b].T>0).astype(np.uint64)<<np.uint64(b)
  w=(counts>0).sum(1).T.astype(np.uint16);h0,h1=masks(prof,w)
  mirrors=[np.flatnonzero((owner!=r)&(rng.random(n)<.5)) for r in range(g)];st=State(owner,mirrors,0,g)
  for e in range(5):
   for b in range(nb):
    batch=[(np.flatnonzero(counts[b,r]),counts[b,r,counts[b,r]>0]) for r in range(g)];st.step(batch)
    actual_f,actual_g,_=st.history[-1];expected=actual_g.copy()
    for r in range(g):
     h=h0 if e==0 else h1;ix=mirrors[r];hit=ix[((h[ix,r]>>np.uint64(b))&np.uint64(1))!=0];expected[r]-=np.bincount(owner[hit],minlength=g)
    assert np.array_equal(actual_f,expected),(g,nb,trial,e,b)
  st.finish()
(ROOT/'cache_test.json').write_text(json.dumps(dict(success=True,cases=24,oracle='independent per-occurrence strict-s0 State version/counter replay, all 5 epochs'),indent=2))
print('PASS 24 randomized strict-s0 cache oracle cases')
