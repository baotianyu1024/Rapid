"""Independent enumeration of traffic after C++ owner optimization, including bit 63."""
import ctypes
from pathlib import Path
import unittest
import numpy as np

class PriceTest(unittest.TestCase):
 def test_hot_first_materialization(self):
  import tempfile,json,subprocess,sys
  from measure import measure
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);case=root/'base';(case/'scheduled').mkdir(parents=True)
   cfg=dict(ranks=4,batches=3,batch_size=32,dense=1,table_lengths=[2,1])
   (case/'config.json').write_text(json.dumps(cfg));rng=np.random.default_rng(4)
   for r in range(4):
    rows=np.column_stack((np.arange(96)+r*96,rng.integers(0,100,96),rng.integers(0,32,(96,3)))).astype('<u4')
    rows.tofile(case/'scheduled'/f'dataset_{r}.bin')
   _,counts=measure([case/'scheduled'/f'dataset_{r}.bin' for r in range(4)],4,32,3,[2,1],1,True)
   np.savez(case/'cps_counts.npz',**counts)
   subprocess.run([sys.executable,str(Path(__file__).with_name('joint_price_owners.py')),'--case',str(case),'--cache',str(root/'cache'),'--output',str(root/'out'),'--hot-first','--hot-threshold','4','--passes','2','--materialize'],check=True,stdout=subprocess.DEVNULL)
   result=json.loads((root/'out/complete.json').read_text());self.assertTrue(result['independent_counts_verified']);self.assertTrue(result['all_records_inverse_verified'])
 def test_exact_counts_and_descent(self):
  fn=ctypes.CDLL(str(Path(__file__).with_name('joint_price_owners.so').resolve())).refine
  fn.restype=ctypes.c_long;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*10+[ctypes.c_double]*5+[ctypes.c_int]*2
  rng=np.random.default_rng(1)
  for nb in (3,16,64):
   n,g=200,4;presence=rng.random((n,nb,g))<.3
   owners=np.zeros(n,dtype=np.int16);powers=np.uint64(1)<<np.arange(nb,dtype=np.uint64)
   profile=(presence*powers[None,:,None]).sum(axis=1,dtype=np.uint64);weight=presence.sum(axis=2,dtype=np.uint16)
   def count():
    receive=np.zeros((nb,g),dtype=np.int64);send=receive.copy()
    for i in range(n):
     for b in range(nb):
      for r in range(g):
       if presence[i,b,r] and r!=owners[i]:receive[b,r]+=1;send[b,owners[i]]+=1
    return receive,send
   receive,send=count();rd=receive.copy();sd=send.copy();storage=np.bincount(owners,minlength=g).astype(np.int64)
   total0=receive.sum(axis=1);norm=total0.astype(float)/g
   def objective():
    return np.mean(np.sum(2*(receive/norm[:,None]-1)**2+(send/norm[:,None]-1)**2,axis=1))+30*np.sum((storage/storage.mean()-1)**2)+.2*np.mean(receive.sum(axis=1)/norm)
   before=objective()
   moved=fn(n,g,nb,*[x.ctypes.data_as(ctypes.c_void_p) for x in [owners,profile,weight,rd,sd,receive,send,storage,norm,total0]],2.,1.,30.,.2,.2,0,8)
   self.assertGreater(moved,0);self.assertLess(objective(),before)
   r,s=count();np.testing.assert_array_equal(r,receive);np.testing.assert_array_equal(s,send)
   np.testing.assert_array_equal(r,rd);np.testing.assert_array_equal(s,sd)
   np.testing.assert_array_equal(storage,np.bincount(owners,minlength=g))
   self.assertTrue(np.all(receive.sum(axis=1)<=total0*1.2))
if __name__=='__main__':unittest.main()
