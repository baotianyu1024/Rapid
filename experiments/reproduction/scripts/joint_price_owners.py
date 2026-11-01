"""Batch-exact joint owner coordinate descent with congestion prices; no mirrors."""
import argparse, ctypes, json, time, hashlib, resource
from pathlib import Path
import numpy as np
from refine_balanced_owners import build_cache, save, materialize
from measure import summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
 p.add_argument('--allow-reduction-tables',action='store_true',help='Optimize raw Unique counts; permit final ACP selection of Reduction tables. Does not optimize the mixed ACP objective.')
 for key,val in [('forward',2.),('backward',1.),('storage',30.),('penalty',.2),('budget',.05)]:p.add_argument('--'+key,type=float,default=val)
 p.add_argument('--uncompressed-maps',action='store_true');p.add_argument('--guard-case');p.add_argument('--library');p.add_argument('--hot-first',action='store_true');p.add_argument('--hot-threshold',type=int,default=0);p.add_argument('--passes',type=int,default=4);p.add_argument('--resume-from');p.add_argument('--materialize',action='store_true');a=p.parse_args()
 case,cache,out=[Path(x).resolve() for x in (a.case,a.cache,a.output)];out.mkdir(exist_ok=False)
 cfg=json.loads((case/'config.json').read_text());g,nb=cfg['ranks'],cfg['batches']
 assert 2<=g<=64 and 1<=nb<=64
 assert min(a.forward,a.backward,a.storage,a.penalty,a.budget)>=0 and a.passes>0
 if a.resume_from:
  assert json.loads((Path(a.resume_from)/'refinement.json').read_text())['base_case']==str(case)
 start=time.perf_counter();build_cache(case,cache,cfg)
 tables=[];storage=np.zeros(g,dtype=np.int64);before=storage.copy()
 for t in range(len(cfg['table_lengths'])):
  ids=np.load(cache/f'{t}.ids.npy',mmap_mode='r');owners=np.load(Path(a.resume_from)/f'{t}.owners.npy') if a.resume_from else (ids%g).astype(np.int16)
  tables.append(dict(ids=ids,owners=owners,profiles=np.asarray(np.load(cache/f'{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64),weights=np.load(cache/f'{t}.weights.npy',mmap_mode='r')))
  before+=np.bincount(ids%g,minlength=g);storage+=np.bincount(owners,minlength=g)
 base=np.load(case/'cps_counts.npz');assert a.allow_reduction_tables or np.all(base['receive'].mean(axis=(0,2))<cfg['batch_size']*(g-1)), 'Only all-Unique traffic is supported; opt in to raw-Unique optimization with final ACP reporting'
 cfg=dict(cfg,allow_reduction_tables=a.allow_reduction_tables,owner_optimization_objective='raw Unique vector traffic; final ACP selected separately')
 initial=np.load(Path(a.resume_from)/'predicted_counts.npz') if a.resume_from else base
 rd=initial['receive'].astype(np.int64);sd=initial['send'].astype(np.int64);recv=rd.sum(axis=1);send=sd.sum(axis=1)
 total0=base['receive'].sum(axis=(1,2)).astype(np.int64);norm=total0.astype(float)/g
 library=Path(a.library).resolve() if a.library else Path(__file__).with_name('joint_price_guarded.so' if a.guard_case else 'joint_price_owners.so')
 lib=ctypes.CDLL(str(library));fn=lib.refine_guarded if a.guard_case else lib.refine
 fn.restype=ctypes.c_long;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*10+[ctypes.c_double]*5+[ctypes.c_int]*2
 ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
 guard_args=[]
 if a.guard_case:
  guard=Path(a.guard_case);gj=json.loads((guard/'refinement.json').read_text());assert gj['base_case']==str(case)
  gc=np.load(guard/'predicted_counts.npz');gr=gc['receive'].sum(1);gs=gc['send'].sum(1)
  fcap=gr.max(1).astype(float);bcap=gs.max(1).astype(float)
  fratio=float((gr.max(1)/gr.mean(1)).max());bratio=float((gs.max(1)/gs.mean(1)).max())
  lower=min(gj['after_storage']);upper=max(gj['after_storage'])
  guard_args=[ptr(fcap),ptr(bcap),fratio,bratio,lower,upper]
  fn.argtypes += [ctypes.c_void_p]*2+[ctypes.c_double]*2+[ctypes.c_int64]*2
 kernel_seconds=0.;kernel_cpu_seconds=0.
 history=[];opt=time.perf_counter()
 if a.hot_first:
  assert a.hot_threshold>0
  for table in tables:
   mass=table['weights'].sum(axis=1,dtype=np.int32)
   table['hot']=np.flatnonzero(mass>=a.hot_threshold);table['cold']=np.flatnonzero(mass<a.hot_threshold)
 for epoch in range(a.passes):
  moved=0
  # Process hot embeddings across ALL tables before allowing cold storage moves.
  order=sorted(range(len(tables)),key=lambda t:-sd[:,t,:].sum())
  schedule=[(phase,t) for phase in (['hot','cold'] if a.hot_first else [None]) for t in order]
  for phase,t in schedule:
   table=tables[t];r=np.ascontiguousarray(rd[:,t,:]);s=np.ascontiguousarray(sd[:,t,:])
   ix=table[phase] if phase else None
   if ix is not None:
    owners=np.ascontiguousarray(table['owners'][ix]);profiles=np.ascontiguousarray(table['profiles'][ix]);weights=np.ascontiguousarray(table['weights'][ix])
   else:owners,profiles,weights=table['owners'],table['profiles'],table['weights']
   kt=time.perf_counter();kc=time.process_time()
   moved+=fn(len(owners),g,nb,*map(ptr,[owners,profiles,weights,r,s,recv,send,storage,norm,total0]),a.forward,a.backward,a.storage,a.penalty,a.budget,epoch%2,a.hot_threshold,*guard_args)
   kernel_seconds+=time.perf_counter()-kt;kernel_cpu_seconds+=time.process_time()-kc
   if ix is not None:table['owners'][ix]=owners
   rd[:,t,:]=r;sd[:,t,:]=s
  assert np.array_equal(recv.sum(axis=1),send.sum(axis=1))
  row=dict(epoch=epoch,moved=moved,forward=summarize(recv),backward=summarize(send),storage_ratio=float(storage.max()/storage.mean()),traffic_change=float(recv.sum()/total0.sum()-1))
  history.append(row);print(json.dumps(row),flush=True)
  if not moved:break
 compute_end=time.perf_counter();save_start=compute_end
 for t,table in enumerate(tables):np.save(out/f'{t}.owners.npy',table['owners'])
 np.savez_compressed(out/'predicted_counts.npz',receive=rd,send=sd)
 save(out/'config.json',dict(cfg,base_case=str(case)))
 save(out/'refinement.json',dict(base_case=str(case),parameters=vars(a),before_storage=before.tolist(),after_storage=storage.tolist(),after_forward=summarize(recv),after_backward=summarize(send),history=history,optimization_seconds=compute_end-opt,kernel_seconds=kernel_seconds,kernel_cpu_seconds=kernel_cpu_seconds,load_and_prepare_seconds=opt-start,save_arrays_seconds=time.perf_counter()-save_start,total_seconds=time.perf_counter()-start,traffic_change=float(recv.sum()/total0.sum()-1)))
 save(out/'optimizer_sha256.json',{str(library.resolve()):hashlib.sha256(library.read_bytes()).hexdigest(),str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),str(library.with_suffix('.cpp').resolve()):hashlib.sha256(library.with_suffix('.cpp').read_bytes()).hexdigest()})
 if a.materialize:materialize(case,cache,out,cfg,tables,rd,sd,compressed=not a.uncompressed_maps)
 print('DONE',out,flush=True)
if __name__=='__main__':main()
