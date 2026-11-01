import ctypes,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;fn=ctypes.CDLL(str(root/'sample_peak.so')).sample_peak;fn.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]*2+[ctypes.c_int]+[ctypes.c_void_p]*2;fn.restype=ctypes.c_int;ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p);rng=np.random.default_rng(44)
for g in [8,64]:
 for trial in range(6):
  b=32;lengths=np.array([3,2,1],np.int32);width=8;data=rng.integers(0,g*20,size=(g*b,width),dtype=np.uint32);data[:,0]=np.arange(g*b);perm=np.zeros(g*b,np.uint32);fr=np.zeros(g,np.int64);fs=fr.copy()
  def count(records):
   r=np.zeros(g,np.int64);s=r.copy()
   for rank in range(g):
    off=2
    for length in lengths:
     ids=np.unique(records[rank*b:(rank+1)*b,off:off+length]);remote=ids[ids%g!=rank];r[rank]+=len(remote);s+=np.bincount(remote%g,minlength=g);off+=length
   return r,s
  bf,bs=count(data);fn(ptr(data),g,b,width,1,3,ptr(lengths),ptr(perm),2,ptr(fr),ptr(fs));assert np.array_equal(np.sort(perm),np.arange(g*b));rf,rs=count(data[perm]);assert np.array_equal(fr,rf) and np.array_equal(fs,rs)
  assert fr.sum()<=bf.sum() and fr.max()<=bf.max() and fs.max()<=bs.max() and (fr+fs).max()<=(bf+bs).max()
(root/'sample_kernel_test.json').write_text(json.dumps(dict(success=True,cases=12,ranks=[8,64],checks=['exact permutation','independent table-scoped vector counts','total and directional/combined peak caps']),indent=2));print('PASS: 12 sample-swap kernel tests')
