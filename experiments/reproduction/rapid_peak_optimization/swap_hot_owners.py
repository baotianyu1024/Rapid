import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'scripts'))
from measure import summarize
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--passes',type=int,default=2);a=p.parse_args();a.case=a.case.resolve();a.cache=a.cache.resolve();a.out=a.out.resolve();a.out.mkdir(exist_ok=False)
cfg=json.load(open(a.case/'config.json'));g=cfg['ranks'];nb=cfg['batches'];nt=len(cfg['table_lengths']);z=np.load(a.resume/'predicted_counts.npz' if a.resume else a.case/'cps_counts.npz');rd=z['receive'].astype(np.int64);sd=z['send'].astype(np.int64);recv=rd.sum(1);send=sd.sum(1);norm=recv.mean(1);start=time.perf_counter()
profiles=[];weights=[];owners=[];offsets=[0];cold=[];hot=[];heat=[];storage=np.zeros(g,np.int64)
for t in range(nt):
 ids=np.load(a.cache/f'{t}.ids.npy',mmap_mode='r');pr=np.asarray(np.load(a.cache/f'{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64);w=np.asarray(np.load(a.cache/f'{t}.weights.npy',mmap_mode='r'));o=np.load(a.resume/f'{t}.owners.npy') if a.resume else (ids%g).astype(np.int16)
 mass=w.sum(1);hot.append(np.flatnonzero(mass>=16).astype(np.int64)+offsets[-1]);heat.append(mass[mass>=16]);cold.append(np.flatnonzero(mass==1).astype(np.int64)+offsets[-1]);profiles.append(pr);weights.append(w);owners.append(o);storage+=np.bincount(o,minlength=g);offsets.append(offsets[-1]+len(ids))
offsets=np.array(offsets,np.int64);order=np.concatenate(cold);capacity=np.full(g,int(offsets[-1])//g,np.int64);capacity[:int(offsets[-1])%g]+=1
hotorder=np.concatenate(hot);heat=np.concatenate(heat);hotorder=hotorder[np.argsort(-heat.astype(np.int64),kind='stable')];lib=ctypes.CDLL(str(ROOT/'swap_owners_sparse.so'));fn=lib.owner_swaps;fn.restype=ctypes.c_long;fn.argtypes=[ctypes.c_long]*2+[ctypes.c_int]*3+[ctypes.c_void_p]*11+[ctypes.c_int];ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
pp=np.array([x.ctypes.data for x in profiles],np.uintp);pw=np.array([x.ctypes.data for x in weights],np.uintp);po=np.array([x.ctypes.data for x in owners],np.uintp)
moved=fn(len(order),len(hotorder),nt,g,nb,*map(ptr,[offsets,order,hotorder,pp,pw,po,rd,sd,recv,send,norm]),a.passes);assert np.array_equal(sum((np.bincount(o,minlength=g) for o in owners),np.zeros(g,np.int64)),storage);assert np.array_equal(recv.sum(1),send.sum(1))
for t,o in enumerate(owners):np.save(a.out/f'{t}.owners.npy',o)
np.savez_compressed(a.out/'predicted_counts.npz',receive=rd,send=sd);(a.out/'config.json').write_text(json.dumps(dict(cfg,base_case=str(a.case)),indent=2))
result=dict(base_case=str(a.case),parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},after_storage=storage.tolist(),after_forward=summarize(recv),after_backward=summarize(send),moved=moved,seconds=time.perf_counter()-start)
(a.out/'refinement.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(dict(moved=moved,mean=float(recv.mean()),recv_peak=float(recv.max(1).mean()),send_peak=float(send.max(1).mean()),combined_peak=float((recv+send).max(1).mean()),storage_min=int(storage.min()),storage_max=int(storage.max()))),flush=True)
