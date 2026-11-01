import ctypes,unittest
from pathlib import Path
import numpy as np
class ExactTest(unittest.TestCase):
 def test_versions(self):
  rng=np.random.default_rng(26);root=Path(__file__).parent
  funcs=[]
  for name in ['joint_price_owners_reference.so','joint_price_owners.so','joint_price_vector.so']:
   fn=ctypes.CDLL(str((root/name).resolve())).refine;fn.restype=ctypes.c_long;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*10+[ctypes.c_double]*5+[ctypes.c_int]*2;funcs.append(fn)
  for g,nb in [(3,1),(8,16),(8,64),(32,16)]:
   for trial in range(10):
    n=200;presence=rng.random((n,nb,g))<(.05+.5*rng.random());owner=rng.integers(0,g,n,dtype=np.int16);profile=(presence*(np.uint64(1)<<np.arange(nb,dtype=np.uint64))[None,:,None]).sum(1,dtype=np.uint64);weights=presence.sum(2,dtype=np.uint16)
    rd=np.zeros((nb,g),dtype=np.int64);sd=rd.copy()
    for i in range(n):
     remote=presence[i].copy();remote[:,owner[i]]=False;rd+=remote;sd[:,owner[i]]+=remote.sum(1)
    storage=np.bincount(owner,minlength=g).astype(np.int64);total=rd.sum(1);norm=total.astype(float)/g
    outputs=[]
    for fn in funcs:
     arrays=[x.copy() for x in [owner,profile,weights,rd,sd,rd,sd,storage,norm,total]]
     moved=fn(n,g,nb,*[x.ctypes.data_as(ctypes.c_void_p) for x in arrays],8.,32.,60.,.5,.05,trial%2,16)
     outputs.append((moved,arrays))
    for moved,arrays in outputs[1:]:
     self.assertEqual(moved,outputs[0][0])
     for a,b in zip(arrays,outputs[0][1]):np.testing.assert_array_equal(a,b)
if __name__=='__main__':unittest.main()
