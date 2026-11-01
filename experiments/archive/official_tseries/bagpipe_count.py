"""Execute upstream OracleCacher.train_addition on CPU; count protocol payloads.
No RPC/network/training is executed. Ring AllReduce cost is an explicit model.
"""
import argparse, ast, contextlib, hashlib, json, subprocess, time, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--lookahead',type=int,default=200);a=p.parse_args()
if (a.out/"result.json").exists():
 print("Existing completed official run",a.out);sys.exit(0)
a.out.mkdir(parents=True,exist_ok=True);v=Path(__file__).resolve().parents[2];source=v/'sources/BagPipe';src=source/'BagPipe/oraclecacher.py';cfg=json.loads((a.data/'input.json').read_text());x=np.load(a.data/'sparse.npy',mmap_mode='r');offset=cfg['active_id_offsets'];lens=cfg['lengths'];cols=np.cumsum([0]+lens);ranks=cfg['ranks'];b=cfg['local_batch'];nb=cfg['global_batches'];steps=nb*a.epochs
dims=cfg['dimensions']
# Reuse epoch arrays (only ID lists, not duplicate full multi-hot data).
uni=[]
for k in range(nb):
 u=[]
 for t in range(len(lens)):
  ids,c=np.unique(x[k*ranks*b:(k+1)*ranks*b,cols[t]:cols[t+1]],return_counts=True);ids=ids.astype(np.int64)-offset[t];u.append((ids,ids[c==1]))
 uni.append(u)
def batches():
 dense=np.zeros((ranks*b,1),np.float32);target=np.zeros((ranks*b,1),np.float32)
 for k in range(steps):
  z=x[(k%nb)*ranks*b:(k%nb+1)*ranks*b]
  sparse=[z[:,cols[t]:cols[t+1]].astype(np.int64)-offset[t] for t in range(len(lens))]
  yield dense,sparse,uni[k%nb],target
 empty=[np.empty((0,l),np.int64) for l in lens];u=[(np.empty(0,np.int64),np.empty(0,np.int64)) for l in lens]
 # Drain the original oracle's buffered real batches; sentinels are never counted.
 for _ in range(a.lookahead+1):yield dense[:0],empty,u,target[:0]
records=[];last=-1;worker_pulls=0;current=None
class RPC:
 def init_rpc(self,*args,**kw):pass
 def rpc_async(self,name,fn,args):
  global last,worker_pulls,current
  d=args[0];k=d['batch_number']
  if k>=steps:return
  tid=int(name.rsplit('_',1)[1]);pf=d['prefetch'];tr=d['train']
  if tid==0:
   assert k==len(records)
   admissions=sum(len(z) for z in pf['prefetch_list'])
   sync=sum(len(z) for z in tr['sync_now'])+sum(len(z) for z in tr['sync_later'])
   admission_bytes=sum(len(z)*dims[t]*4 for t,z in enumerate(pf['prefetch_list']));sync_bytes=sum((len(tr['sync_now'][t])+len(tr['sync_later'][t]))*dims[t]*4 for t in range(len(dims)))
   current=dict(ps_pull_bytes=0,ps_writeback_bytes=admission_bytes,gradient_allreduce_bytes=2*(ranks-1)*sync_bytes,step=k,ps_pull_vectors=0,ps_writeback_vectors_including_terminal_flush=admissions,gradient_allreduce_vectors=2*(ranks-1)*sync,synchronized_ids=sync,admissions=admissions)
  # Exact worker-side no_sync pruning from upstream bagcache.py.
  for t,ids in enumerate(pf['prefetch_list']):
   nos=pf['no_sync'][t];local=np.unique(tr['train_data']['sparse_vector'][t].numpy());other=nos[~np.isin(nos,local,assume_unique=True)];pruned=ids[~np.isin(ids,other,assume_unique=True)];current['ps_pull_vectors']+=len(pruned);current['ps_pull_bytes']+=len(pruned)*dims[t]*4
  if tid==ranks-1:records.append(current)
obj=SimpleNamespace(worker_name='oracle',worker_id=0,world_size=ranks+2,num_trainers=ranks,trainer_prefix='trainer',update_queue=None,verbose=False,ragged=True,lookahead_value=a.lookahead,train_ld=iter(batches()),ln_emb_len=len(lens),ln_emb=np.diff(offset),batch_size_on_each_machine=b,local_cache_ttl={t:np.full(offset[t+1]-offset[t],-1,np.int64) for t in range(len(lens))},latest_tracker={t:np.full(offset[t+1]-offset[t],-1,np.int64) for t in range(len(lens))})
for n in ['batch_queue_dense','batch_queue_sparse','batch_queue_target','batch_number_queue','batch_emb_union','batch_sync_now','batch_sync_later','batch_once']:setattr(obj,n,[])
tree=ast.parse(src.read_text());cls=next(z for z in tree.body if isinstance(z,ast.ClassDef) and z.name=='OracleCacher');fn=next(z for z in cls.body if isinstance(z,ast.FunctionDef) and z.name=='train_addition');mod=ast.Module(body=[fn],type_ignores=[]);ns=dict(np=np,torch=torch,time=time.time,rpc=RPC());exec(compile(mod,str(src),'exec'),ns);start=time.perf_counter()
with open(a.out/'oracle.log','w') as log,contextlib.redirect_stdout(log):ns['train_addition'](obj)
assert len(records)==steps
keys=['ps_pull_vectors','ps_writeback_vectors_including_terminal_flush','gradient_allreduce_vectors'];totals={k:sum(d[k] for d in records) for k in keys};total=sum(totals.values())
result=dict(method='BagPipe',measurement='Upstream CPU oracle execution + explicit protocol accounting, not measured network',source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),oracle_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),rows=cfg['rows'],epochs=a.epochs,steps=steps,ranks=ranks,dim="per-table official",lookahead=a.lookahead,**totals,total_vectors_with_flush=total,total_embedding_bytes=sum(sum(d[k] for k in ['ps_pull_bytes','ps_writeback_bytes','gradient_allreduce_bytes']) for d in records),dimensions=dims,seconds=time.perf_counter()-start,cache_entries_per_worker_upper_bound=cfg['active_embeddings'],notes=['Original train_addition function executed unchanged via AST, RPC replaced with local event capture.','AllReduce counts use ring sent bytes: 2*(P-1)*synchronized IDs across all ranks; dense gradients and IDs excluded.','Every admitted cache lifetime has one eventual PS writeback; all unfinished lifetimes flushed at termination. This is modeled terminal accounting, not an executed GPU cache cleanup.','Empty sentinel batches drain lookahead buffer; only original real steps counted.',f'Configured lookahead={a.lookahead}; upstream default=200. No common cache-memory budget is enforced; upper bound excludes gradient/optimizer storage.'])
(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(a.out/'steps.json').write_text(json.dumps(records)+'\n');print(json.dumps(result,indent=2))
