import tempfile,subprocess,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;rng=np.random.default_rng(3)
for g,nb in [(8,64),(64,8)]:
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp);c=p/'cache';c.mkdir();n=g*20;prof=np.zeros((n,g),np.uint64)
  for i in range(n):
   for j in range(1 if i<n//2 else 32):prof[i,rng.integers(g)]|=np.uint64(1)<<np.uint64(rng.integers(nb))
  w=np.array([((prof>>np.uint64(b))&np.uint64(1)).sum(1) for b in range(nb)],np.uint16).T.copy()
  np.save(c/'0.ids.npy',np.arange(n,dtype=np.uint32));np.save(c/'0.profiles.npy',prof);np.save(c/'0.weights.npy',w)
  def count(o):
   r=np.zeros((nb,1,g),np.int64);s=r.copy()
   for i in range(n):
    for rank in range(g):
     if o[i]!=rank:
      for b in range(nb):
       if (int(prof[i,rank])>>b)&1:r[b,0,rank]+=1;s[b,0,o[i]]+=1
   return r,s
  old=np.arange(n)%g;r,s=count(old);before_r=r.copy();before_s=s.copy();np.savez(p/'cps_counts.npz',receive=r,send=s);(p/'config.json').write_text(json.dumps(dict(ranks=g,batches=nb,rows=g*nb*32,batch_size=32,table_lengths=[1],dense=1)))
  subprocess.run([sys.executable,str(ROOT/'swap_hot_budget.py'),'--case',str(p),'--cache',str(c),'--out',str(p/'out')],check=True)
  own=np.load(p/'out/0.owners.npy');r,s=count(own);z=np.load(p/'out/predicted_counts.npz');assert np.array_equal(r,z['receive']) and np.array_equal(s,z['send']);assert np.array_equal(np.bincount(own,minlength=g),np.bincount(old,minlength=g));assert np.all(r.max(2)<=before_r.max(2)) and np.all(s.max(2)<=before_s.max(2)) and np.all((r+s).max(2)<=(before_r+before_s).max(2)) and np.all(r.sum(2)<=np.floor(before_r.sum(2)*1.005))
(ROOT/'owner_swaps_budget_test.json').write_text(json.dumps(dict(success=True,ranks=[8,64],independent_counters=True,exact_unchanged_storage=True),indent=2));print('OWNER SWAPS TEST PASS')
