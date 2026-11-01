import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;lib=ctypes.CDLL(str(root/'sample_group_peak.so'));fn=lib.sample_group_peak;fn.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*4+[ctypes.c_double]+[ctypes.c_void_p]*3;fn.restype=ctypes.c_int;ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
LAST_COUNTS={}
def grouped(data,n,width,m,g,b,owners,assignment,rounds,balance,group_keys,budget=0):
 data=np.ctypeslib.as_array(ctypes.cast(data,ctypes.POINTER(ctypes.c_uint32)),shape=(n*width,)).reshape(n,width);owners=np.ctypeslib.as_array(ctypes.cast(owners,ctypes.POINTER(ctypes.c_int16)),shape=(m,));assignment=np.ctypeslib.as_array(ctypes.cast(assignment,ctypes.POINTER(ctypes.c_int32)),shape=(n,));nr=n//g;nb=nr//b;assert m*g<2**32;assert np.array_equal(assignment,np.repeat(np.arange(g),nr));rounds=max(rounds,8)
 LAST_COUNTS.clear();LAST_COUNTS.update(receive=np.zeros((nb,g),np.int64),send=np.zeros((nb,g),np.int64))
 def work(k):
  indices=(np.arange(g)[:,None]*nr+k*b+np.arange(b)[None,:]).ravel();values=data[indices];rec=np.empty((g*b,width+1),np.uint32);rec[:,0]=0;rec[:,1:]=values*np.uint32(g)+owners[values].astype(np.uint32);keys=np.asarray(group_keys[indices],np.uint64);out=np.empty(g*b,np.uint32);f=np.zeros(g,np.int64);s=f.copy();moves=fn(ptr(rec),ptr(keys),g,b,width+1,rounds,budget,*map(ptr,[out,f,s]));assert np.array_equal(np.sort(out),np.arange(g*b));assignment[indices[out]]=np.repeat(np.arange(g),b);LAST_COUNTS['receive'][k]=f;LAST_COUNTS['send'][k]=s;print('group exact batch',k,'swaps',moves,'peak',int((f+s).max()),flush=True);return moves
 with ThreadPoolExecutor(max_workers=8) as pool:return sum(pool.map(work,range(nb)))
