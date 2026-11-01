from pathlib import Path
import json,hashlib
import numpy as np

def save(path,v):
 Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(v,indent=2))
def read(path):return json.loads(Path(path).read_text())
def inverse_chain(case):
 case=Path(case);chain=[];cur=case
 while True:
  cfg=read(cur/'config.json');chain.append(cur)
  if not cfg.get('base_case'):break
  cur=Path(cfg['base_case'])
 g=cfg['ranks'];nt=len(cfg['table_lengths']);placement=[[[] for _ in range(g)] for _ in range(nt)]
 for r in range(g):
  for line in (cur/f'placement/table_{r}').read_text().splitlines():
   t,f=map(int,line.split(':'));placement[t][r].append(f)
 inverse=[]
 for t in range(nt):
  lut=np.full(max(len(v)*g for v in placement[t]),0xffffffff,np.uint32)
  for r in range(g):
   ids=np.array(sorted(placement[t][r]),np.uint32);lut[np.arange(len(ids))*g+r]=ids
  for stage in reversed(chain[:-1]):
   if not (stage/f'table_{t}_remap.npz').exists():continue
   z=np.load(stage/f'table_{t}_remap.npz');assert np.all(lut[z['old']]!=0xffffffff);nxt=np.full(int(z['new'].max())+1,0xffffffff,np.uint32);nxt[z['new']]=lut[z['old']];lut=nxt
  inverse.append(lut)
 return inverse,cur

def audit_records(case,source=None):
 case=Path(case);cfg=read(case/'config.json');g=cfg['ranks'];b=cfg['batch_size'];nb=cfg['batches'];width=1+cfg['dense']+sum(cfg['table_lengths']);inverse,base=inverse_chain(case)
 raw=np.memmap(source or read(base/'config.json')['input'],dtype='<u4',mode='r').reshape(-1,width)
 dest=[np.memmap(case/f'scheduled/dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width) for r in range(g)]
 assert all(len(v)==nb*b for v in dest)
 def sort(x):return np.sort(np.ascontiguousarray(x).view(f'V{4*width}').reshape(-1))
 for k in range(nb):
  x=np.concatenate([v[k*b:(k+1)*b] for v in dest]);col=1+cfg['dense']
  for t,l in enumerate(cfg['table_lengths']):x[:,col:col+l]=inverse[t][x[:,col:col+l]];col+=l
  assert np.array_equal(sort(x),sort(raw[k*g*b:(k+1)*g*b])),('raw records/batch mismatch',case,k)
 return dict(success=True,rows=nb*g*b,batches=nb,all_record_bytes_and_multiplicities=True,original_global_batch_preserved=True)

def export(case,out):
 case=Path(case);out=Path(out);out.mkdir(exist_ok=False);cfg=read(case/'config.json');g=cfg['ranks'];inv,_=inverse_chain(case);storage=np.zeros(g,np.int64)
 for t,lut in enumerate(inv):
  physical=np.flatnonzero(lut!=0xffffffff).astype(np.uint32);ids=lut[physical];order=np.argsort(ids);ids=ids[order];physical=physical[order];own=(physical%g).astype(np.int16);storage+=np.bincount(own,minlength=g)
  np.savez(out/f'table_{t}.npz',original_ids=ids,physical_ids=physical,owners=own)
 save(out/'manifest.json',dict(ranks=g,primary_counts=storage.tolist(),source=str(case)))

def traffic(fr,fs,br,bs,processed_rows,full=None):
 full=np.ones(len(fr),bool) if full is None else full;c=fr+br;mean=c.mean(0)
 return dict(total_vectors=int(c.sum()),GiB128=float(c.sum()*512/2**30),vectors_per_8192=float(c.sum()/processed_rows*8192),mean_step_peak=float(c[full].max(1).mean()),absolute_peak=int(c.max()),phase_peak=float((np.maximum(fr[full].max(1),fs[full].max(1))+np.maximum(br[full].max(1),bs[full].max(1))).mean()),rank_means=mean.tolist(),rank_cv=float(mean.std()/mean.mean()),rank_max_mean=float(mean.max()/mean.mean()),steps=len(c),full_steps=int(full.sum()))
