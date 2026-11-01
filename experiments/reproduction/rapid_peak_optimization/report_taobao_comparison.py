"""Audit complete-batch records and report Taobao communication with explicit scopes."""
import argparse,json,csv,sys
from pathlib import Path
import numpy as np
X=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2))
def metric(fr,fs,br,bs,rows,full):
 c=fr+br;rankmeans=c.mean(0)
 return dict(total_vectors=int(c.sum()),total_GiB=float(c.sum()*512/2**30),vectors_per_8192=float(c.sum()/rows*8192),steps=len(c),full_steps=int(full.sum()),mean_step_peak_receive=float(c[full].max(1).mean()),mean_step_peak_send=float((fs+bs)[full].max(1).mean()),absolute_peak_receive=int(c.max()),worst_rank_mean_receive=float(rankmeans.max()),phase_bottleneck=float((np.maximum(fr[full].max(1),fs[full].max(1))+np.maximum(br[full].max(1),bs[full].max(1))).mean()),forward_peak=float(fr[full].max(1).mean()),backward_peak=float(br[full].max(1).mean()),rank_mean_max_over_mean=float(rankmeans.max()/rankmeans.mean()),rank_mean_cv=float(rankmeans.std()/rankmeans.mean()),rank_totals=c.sum(0).tolist())
reports={};flat=[]
for case in a.cases:
 assert read(case/'complete.json')['success']
 cfg=read(case/'experiment.json');g=cfg['ranks'];nb=cfg['batches'];n=cfg['rows'];b=8192;opt=case/'optimized';final=opt/'peak_samples';cache=opt/'cache';model=case/'baseline/rapid_schedule'
 assert read(cache/'verification.json')['success']
 # Compose every inverse embedding map back to source IDs.
 placement=[[[] for _ in range(g)] for _ in range(4)]
 for r in range(g):
  for line in (model/f'placement/table_{r}').read_text().splitlines():
   t,f=map(int,line.split(':'));placement[t][r].append(f)
 inverse=[]
 for t in range(4):
  size=max((len(v)*g for v in placement[t]),default=0);lut=np.full(size,0xffffffff,np.uint32)
  for r in range(g):
   ids=np.array(sorted(placement[t][r]),np.uint32);lut[np.arange(len(ids))*g+r]=ids
  for stage in [case/'baseline/rapid_owner_base',opt/'symmetric',opt/'cold_repair',opt/'hot_guard',opt/'hot_budget']:
   z=np.load(stage/f'table_{t}_remap.npz');old=z['old'];new=z['new'];assert np.all(lut[old]!=0xffffffff)
   nxt=np.full(int(new.max())+1,0xffffffff,np.uint32);nxt[new]=lut[old];lut=nxt
  inverse.append(lut)
 raw=np.memmap(case/'data/input.bin',dtype='<u4',mode='r').reshape(n,6)
 scheduled=[np.memmap(final/f'scheduled/dataset_{r}.bin',dtype='<u4',mode='r').reshape(nb*b,6) for r in range(g)]
 def sorted_records(x):return np.sort(np.ascontiguousarray(x).view('V24').reshape(-1))
 for k in range(nb):
  restored=np.concatenate([v[k*b:(k+1)*b] for v in scheduled])
  for t in range(4):restored[:,2+t]=inverse[t][restored[:,2+t]]
  assert np.array_equal(sorted_records(restored),sorted_records(raw[k*g*b:(k+1)*g*b])),f'batch {k} records differ'
 print('full raw-record batch audit passed',g,flush=True)
 z=np.load(cache/'counts.npz');fr,fs,br,bs=[z[k] for k in ['fr','fs','br','bs']]
 methods={'rapid_latest':metric(fr,fs,br,bs,n*5,np.ones(nb*5,bool))}
 # Include no-mirror RAPID to disclose effect of extension.
 methods['rapid_latest_no_mirrors']=metric(bs,br,br,bs,n*5,np.ones(nb*5,bool))
 primary=np.zeros(g,np.int64)
 for t in range(4):primary+=np.bincount(np.load(final/f'{t}.owners.npy'),minlength=g)
 methods['rapid_latest']['primary_counts']=primary.tolist();methods['rapid_latest']['mirror_counts']=read(cache/'result.json')['mirror_counts']
 sparse=np.load(case/'data/sparse.npy',mmap_mode='r')
 for tag,part,traffic in [('gmp_native',case/'gmp',case/'gmp_traffic'),('gmp_fixed_batch',case/'gmp_fixed/partition',case/'gmp_fixed/traffic')]:
  meta=read(traffic/'het_gmp_s0.json');z=np.load(traffic/'het_gmp_s0.npz');f=z['forward'];back=z['backward'];steps=meta['steps_per_epoch'];assignment=np.load(part/'assignment.npy');owner=np.load(part/'owner.npy');rankrows=[np.flatnonzero(assignment==r) for r in range(g)]
  # Every s=0 remote gradient independently reconstructed from raw sparse samples.
  independent=np.zeros((steps,g,g),np.int64)
  for k in range(steps):
   for r in range(g):
    ids=np.unique(sparse[rankrows[r][k*b:(k+1)*b]].ravel());own=owner[ids];independent[k,r]=np.bincount(own[own!=r],minlength=g)
  assert np.array_equal(back,np.tile(independent,(5,1,1)))
  assert meta['pending_updates']==0 and meta['processed_rows']==n*5
  if tag=='gmp_fixed_batch':
   for k in range(nb):assert np.array_equal(np.bincount(assignment[k*g*b:(k+1)*g*b],minlength=g),np.full(g,b))
  full=np.arange(len(f))%steps<min(meta['rank_rows'])//b
  methods[tag]=metric(f.sum(2),f.sum(1),back.sum(1),back.sum(2),n*5,full)
  methods[tag].update(primary_counts=meta['primary_counts'],mirror_counts=meta['mirror_counts'],rank_rows=meta['rank_rows'])
 rapid=methods['rapid_latest'];deltas={}
 for tag in ['gmp_native','gmp_fixed_batch']:
  deltas[tag]={key:100*(1-rapid[key]/methods[tag][key]) for key in ['total_vectors','mean_step_peak_receive','phase_bottleneck']}
 result=dict(config=cfg,input_sha256=read(case/'data/input.json')['input_sha256'],methods=methods,rapid_reduction_percent=deltas,verification=dict(all_source_records_and_original_global_batches_verified=True,cache_independent_version_oracle=True,gmp_gradients_independent_raw_replay=True),notes=['Total counts both forward and backward; each transferred vector counted once, not send+receive twice.','Mean step peak receive = mean over steps of max_rank(forward receive + backward receive).','Native GMP full-step peaks exclude uneven tail steps; all-sample totals include tails.','128 float32 components per vector,512 bytes; excludes metadata and dense AllReduce; CPU replay,not GPU timing.','4,194,304 rows from existing shuffled four-table Taobao,not full26,557,961 rows.','RAPID fixed Criteo-selected parameters,not tuned for Taobao; strict-s0 mirror extension shares GMP capacity.'])
 reports[str(g)]=result
 for name,v in methods.items():flat.append(dict(ranks=g,method=name,**{k:v[k] for k in ['total_vectors','total_GiB','vectors_per_8192','mean_step_peak_receive','phase_bottleneck','absolute_peak_receive','rank_mean_max_over_mean']}))
 save(case/'comparison.json',result)
if (a.out/'comparison.json').exists():
 prior=read(a.out/'comparison.json');prior.update(reports);reports=prior
flat=[dict(ranks=int(g),method=name,**{k:v[k] for k in ['total_vectors','total_GiB','vectors_per_8192','mean_step_peak_receive','phase_bottleneck','absolute_peak_receive','rank_mean_max_over_mean']}) for g,report in sorted(reports.items(),key=lambda kv:int(kv[0])) for name,v in report['methods'].items()]
save(a.out/'comparison.json',reports)
with (a.out/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(json.dumps(flat,indent=2))
