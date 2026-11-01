"""Capacity-exact RAPID owner placement; cold-locality reservations, hot-first service balancing."""
import argparse,ctypes,json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'scripts'))
from measure import summarize
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--forward',type=float,default=1);p.add_argument('--backward',type=float,default=4);p.add_argument('--penalty',type=float,default=1);a=p.parse_args();a.case=a.case.resolve();a.cache=a.cache.resolve();a.out=a.out.resolve();a.out.mkdir(exist_ok=False)
cfg=json.load(open(a.case/'config.json'));g=cfg['ranks'];nb=cfg['batches'];nt=len(cfg['table_lengths']);z=np.load(a.case/'cps_counts.npz');rd=z['unique'].astype(np.int64);sd=np.zeros_like(rd);recv=rd.sum(1);send=np.zeros_like(recv);norm=z['receive'].sum((1,2)).astype(float)/g
profiles=[];weights=[];owners=[];offsets=[0];masses=[];reserve=np.zeros(g,np.int64);start=time.perf_counter()
for t in range(nt):
 pr=np.asarray(np.load(a.cache/f'{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64);w=np.asarray(np.load(a.cache/f'{t}.weights.npy',mmap_mode='r'));mass=w.sum(1,dtype=np.uint16);ix=np.flatnonzero(mass==1)
 if len(ix):reserve+=np.bincount(np.argmax(pr[ix]!=0,axis=1),minlength=g)
 profiles.append(pr);weights.append(w);owners.append(np.full(len(pr),-1,np.int16));masses.append(mass);offsets.append(offsets[-1]+len(pr))
offsets=np.array(offsets,np.int64);mass=np.concatenate(masses);order=np.argsort(-mass.astype(np.int32),kind='stable').astype(np.int64);n=len(order);capacity=np.full(g,n//g,np.int64);capacity[:n%g]+=1;reserve=np.minimum(reserve,capacity);storage=np.zeros(g,np.int64)
lib=ctypes.CDLL(str(ROOT/'capacity_owner.so'));fn=lib.capacity_owner;fn.restype=None;fn.argtypes=[ctypes.c_long]+[ctypes.c_int]*3+[ctypes.c_void_p]*13+[ctypes.c_double]*3
ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p);pp=np.array([x.ctypes.data for x in profiles],dtype=np.uintp);pw=np.array([x.ctypes.data for x in weights],dtype=np.uintp);po=np.array([x.ctypes.data for x in owners],dtype=np.uintp)
print('loaded, running',n,'IDs',g,'ranks',flush=True)
fn(n,nt,g,nb,*map(ptr,[offsets,order,pp,pw,po,rd,sd,recv,send,storage,capacity,reserve,norm]),a.forward,a.backward,a.penalty)
assert np.array_equal(storage,capacity) and np.array_equal(recv.sum(1),send.sum(1)) and np.all(recv>=0)
for t,o in enumerate(owners):assert np.all((o>=0)&(o<g));np.save(a.out/f'{t}.owners.npy',o)
np.savez_compressed(a.out/'predicted_counts.npz',receive=rd,send=sd)
(a.out/'config.json').write_text(json.dumps(dict(cfg,base_case=str(a.case)),indent=2))
result=dict(base_case=str(a.case),parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},method='hard exact primary capacities, reserve exclusive cold IDs locally, assign global hot-to-cold order with exact minibatch service/receiver prices',after_storage=storage.tolist(),after_forward=summarize(recv),after_backward=summarize(send),seconds=time.perf_counter()-start)
(a.out/'refinement.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(dict(mean=float(recv.mean()),recv_peak=float(recv.max(1).mean()),send_peak=float(send.max(1).mean()),combined_peak=float((recv+send).max(1).mean()),storage_min=int(storage.min()),storage_max=int(storage.max()))),flush=True)
