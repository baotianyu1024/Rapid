import tempfile,json,copy
from persistent_fast import *
rng=np.random.default_rng(81)
for g,nb in [(8,64),(64,8)]:
 n=270;owner=rng.integers(0,g,n,dtype=np.int16);mirrors=[np.flatnonzero(owner!=r).astype(np.uint32) for r in range(g)];a=State(owner.copy(),mirrors,0,g);b=copy.deepcopy(a)
 for window in range(4):
  if window:
   new=np.r_[owner,rng.integers(0,g,5,dtype=np.int16)];new[:2]=(new[:2]+1)%g;update_owners(a,new,[0,1]);update_owners(b,new,[0,1]);owner=new;n=len(owner)
   mirrors=[np.sort(rng.choice(np.flatnonzero(owner!=r),40,replace=False)).astype(np.uint32) for r in range(g)];set_mirrors(a,mirrors);set_mirrors(b,mirrors)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'profile_cache').mkdir();prof=np.zeros((n,g),np.uint64);trace=[]
   for k in range(nb):
    batches=[]
    for r in range(g):
     ids=np.sort(rng.choice(n,8,replace=False)).astype(np.uint32);prof[ids,r]|=np.uint64(1)<<np.uint64(k);batches.append((ids,np.ones(len(ids),np.uint16)))
    trace.append(batches)
   np.save(p/'profile_cache/0.profiles.npy',prof);np.save(p/'0.logical_gids.npy',np.arange(n,dtype=np.uint32));np.save(p/'0.owners.npy',owner);(p/'config.json').write_text(json.dumps(dict(ranks=g,batches=nb,table_lengths=[1])))
   ra,aa=replay_window(a,trace);rb,bb=replay_case(b,p)
   for key in aa:assert np.array_equal(aa[key],bb[key]),key
   assert np.array_equal(a.master,b.master)
   for r,ids in enumerate(a.mirrors):assert np.array_equal(a.version[r]==a.master[ids],b.version[r]==b.master[ids])
print('PERSISTENT NATIVE/PYTHON MULTIWINDOW AB PASS')
