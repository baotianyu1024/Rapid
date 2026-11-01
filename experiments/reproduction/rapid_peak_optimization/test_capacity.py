import tempfile,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;rng=np.random.default_rng(39)
for g,nb in [(8,64),(64,8)]:
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp);cache=p/'profile';cache.mkdir();present=[];unique=np.zeros((nb,2,g),np.int64);rd=unique.copy();sd=rd.copy()
  for t,n in enumerate([193,321]):
   pr=rng.random((n,nb,g))<.06;pr[np.arange(n),rng.integers(nb,size=n),rng.integers(g,size=n)]=True
   profile=(pr.astype(np.uint64)*(np.uint64(1)<<np.arange(nb,dtype=np.uint64))[None,:,None]).sum(1);w=pr.sum(2,dtype=np.uint16);np.save(cache/f'{t}.profiles.npy',profile);np.save(cache/f'{t}.weights.npy',w);present.append(pr)
   for i in range(n):
    r=pr[i].copy();unique[:,t]+=r;r[:,i%g]=False;rd[:,t]+=r;sd[:,t,i%g]+=r.sum(1)
  np.savez(p/'cps_counts.npz',unique=unique,receive=rd,send=sd);(p/'config.json').write_text(json.dumps(dict(ranks=g,batches=nb,batch_size=32,rows=g*nb*32,table_lengths=[1,1],dense=1)))
  subprocess.run([sys.executable,str(ROOT/'capacity_owner.py'),'--case',str(p),'--cache',str(cache),'--out',str(p/'out')],check=True,stdout=subprocess.DEVNULL)
  er=np.zeros_like(rd);es=er.copy();st=np.zeros(g,np.int64)
  for t,pr in enumerate(present):
   o=np.load(p/f'out/{t}.owners.npy');st+=np.bincount(o,minlength=g)
   for i in range(len(o)):
    r=pr[i].copy();r[:,o[i]]=False;er[:,t]+=r;es[:,t,o[i]]+=r.sum(1)
  z=np.load(p/'out/predicted_counts.npz');assert np.array_equal(z['receive'],er) and np.array_equal(z['send'],es);assert st.max()-st.min()<=1
(ROOT/'capacity_test.json').write_text(json.dumps(dict(success=True,ranks=[8,64],checks=['independent exact per-table directed communication','perfect integer storage capacities']),indent=2));print('CAPACITY OWNER TEST PASS')
