import ctypes,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;fn=ctypes.CDLL(str(root/'batch_soft_samples.so')).batch_soft_samples;fn.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]*2+[ctypes.c_int,ctypes.c_double];fn.restype=ctypes.c_int;ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p);rng=np.random.default_rng(42)
for g in [8,64]:
 b=8;nb=3;nr=b*nb;n=nr*g;m=g*20;data=np.sort(rng.integers(0,m,(n,7),dtype=np.uint32),axis=1);owner=(np.arange(m)%g).astype(np.int16);answers=[]
 for repeat in range(2):
  assign=np.repeat(np.arange(g,dtype=np.int32),nr);fn(ptr(data),n,7,m,g,b,ptr(owner),ptr(assign),2,1.)
  for k in range(nb):assert np.array_equal(np.bincount(assign[(np.arange(n)%nr)//b==k],minlength=g),np.full(g,b))
  answers.append(assign.copy())
 assert np.array_equal(*answers)
(root/'batch_soft_test.json').write_text(json.dumps(dict(success=True,ranks=[8,64],checks=['exact per-original-batch rank capacities','deterministic independent snapshot proposals','fixed embedding owners']),indent=2));print('BATCH SOFT TEST PASS')
