import tempfile,subprocess,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;rng=np.random.default_rng(2)
for g,nb in [(8,64),(64,8)]:
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp);c=p/'cache';c.mkdir();n=g*20;home=rng.integers(g,size=n);batch=rng.integers(nb,size=n);prof=np.zeros((n,g),np.uint64);prof[np.arange(n),home]=np.uint64(1)<<batch.astype(np.uint64);w=np.zeros((n,nb),np.uint16);w[np.arange(n),batch]=1
  np.save(c/'0.ids.npy',np.arange(n,dtype=np.uint32)*g);np.save(c/'0.profiles.npy',prof);np.save(c/'0.weights.npy',w)
  def count(o):
   r=np.zeros((nb,1,g),np.int64);s=r.copy()
   for i in range(n):
    if o[i]!=home[i]:r[batch[i],0,home[i]]+=1;s[batch[i],0,o[i]]+=1
   return r,s
  r,s=count(np.zeros(n,int));np.savez(p/'cps_counts.npz',receive=r,send=s);(p/'config.json').write_text(json.dumps(dict(ranks=g,batches=nb,rows=g*nb*32,batch_size=32,table_lengths=[1],dense=1)))
  subprocess.run([sys.executable,str(ROOT/'cold_capacity.py'),'--case',str(p),'--cache',str(c),'--out',str(p/'out')],stdout=subprocess.DEVNULL,check=True)
  own=np.load(p/'out/0.owners.npy');r,s=count(own);z=np.load(p/'out/predicted_counts.npz');assert np.array_equal(r,z['receive']) and np.array_equal(s,z['send']);assert np.bincount(own,minlength=g).ptp()<=1
(ROOT/'cold_test.json').write_text(json.dumps(dict(success=True,ranks=[8,64],independent_counters=True),indent=2));print('COLD CAPACITY REPAIR TEST PASS')
