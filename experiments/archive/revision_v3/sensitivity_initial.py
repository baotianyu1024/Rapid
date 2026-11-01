"""Fresh core CPS sensitivity, batch-size and component ablations."""
from pathlib import Path
import subprocess,sys,json,os,time
import numpy as np
from evidence_utils import *
H=Path(__file__).resolve().parents[1];W=Path("/mnt/hugui/longjiu.bty");R=H/'code/reproduction';S=R/'scripts';C=R/'working/cps';sys.path.insert(0,str(S));from measure import measure,summarize_counts
out=H/'results/sensitivity';out.mkdir(exist_ok=True);n=262144;g=8;L=[3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1];raw=np.memmap(W/'data/processed/criteo/day6.bin',dtype='<u4',mode='r').reshape(-1,228)[:n];source=out/'input.bin';
if not source.exists():raw.tofile(source)
env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(H/'dependencies')+':'+str(W/'reproduction/deps'),LD_LIBRARY_PATH=str(W/'reproduction/toolchain/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
def run(name,args):
 cmd=list(map(str,args));save(out/(name+'.command.json'),cmd);t=time.perf_counter()
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 save(out/(name+'.status.json'),dict(exit_code=r.returncode,seconds=time.perf_counter()-t));assert r.returncode==0,name
common=['--dense_count',13,'--table_lengths',*L];hots={};results=read(out/'summary.json')['results'] if (out/'summary.json').exists() else []
def hot(k):
 if k not in hots:
  hots[k]=out/f'hot{k}'
  if not hots[k].exists():run(f'hot{k}',[sys.executable,C/'hot_id_extraction.py','--input',source,'--output',hots[k],'--topk',k,'--max_rows',n,*common])
 return hots[k]
def case(name,beta=.0001,gamma=.2,k=40000,batch=8192,seedrows=0,order=None,hashplace=False,placement=None):
 d=out/name
 if any(v['name']==name for v in results):return d
 d.mkdir(exist_ok=True);nb=n//(g*batch);inp=source
 if order is not None:
  ix=np.arange(n//(g*8192));ix=ix[::-1] if order=='reverse' else np.random.default_rng(order).permutation(ix);inp=d/'profile.bin';raw.reshape(-1,g*8192,228)[ix].tofile(inp)
 place=placement or d/'placement'
 if placement is None:
  if hashplace:
   place.mkdir();files=[(place/f'table_{r}').open('w') for r in range(g)];col=14
   for t,l in enumerate(L):
    ids=np.unique(raw[:,col:col+l]);col+=l
    for r in range(g):files[r].write(''.join(f'{t}:{int(i)}\n' for i in ids[ids%g==r]))
   for f in files:f.close()
  else:
   extra=[]
   if seedrows:
    seed=d/'seed';run(name+'_seed',[sys.executable,S/'build_metis_seed.py','--input',source,'--hot',hot(k),'--output',seed,'--rows',seedrows,'--sample-scope-rows',n,'--ranks',g]);extra=['--seed_assignment',seed/f'seed_{g}.txt']
   run(name+'_placement',[C/'arrange_ids_to_devices','--input_bin',inp,'--hot_ids_dir',hot(k),'--output_dir',place,'--batch_size',8192,'--num_devices',g,'--n_batches',n//(g*8192),'--beta',beta,*extra,*common])
 run(name+'_schedule',[C/'generate_local_dataset','--input_file',source,'--table_files_dir',place,'--output_dir',d/'scheduled','--num_devices',g,'--train_batch_size',batch,'--num_chunks',nb,'--steps_per_chunk',1,'--num_threads',8,'--gamma',gamma,'--swap_algorithm','ratio','--swap_screen_queries_per_gpu',20000000,'--swap_evaluations_per_gpu',500000,*common])
 if placement is not None:(d/'placement').symlink_to(place.resolve(),target_is_directory=True)
 save(d/'config.json',dict(input=str(source),base_case=None,ranks=g,batch_size=batch,batches=nb,rows=n,table_lengths=L,dense=13))
 m,z=measure([d/f'scheduled/dataset_{r}.bin' for r in range(g)],g,batch,nb,L,13,True);np.savez_compressed(d/'counts.npz',**z);save(d/'metrics.json',m)
 storage=[sum(1 for _ in (place/f'table_{r}').open()) for r in range(g)];choose=z['receive'].mean(axis=(0,2))<batch*(g-1);f=np.where(choose[None,:,None],z['receive'],batch*(g-1)).sum(1);b=np.where(choose[None,:,None],z['send'],batch*(g-1)).sum(1)
 result=dict(name=name,beta=beta,gamma=gamma,topk=k,batch=batch,seed_rows=seedrows,seed_fraction=seedrows/n,profile_order=order,storage=storage,storage_max_mean=max(storage)/np.mean(storage),local_hit=m['unique_local_hit_rate'],acp_vectors_per_sample=float(2*f.sum()/n),acp_peak_per_step=float((f+b).max(1).mean()),unique_vectors_per_sample=float(2*z['receive'].sum()/n),audit=audit_records(d));results.append(result);save(d/'initial_metrics.json',result);print(json.dumps(result),flush=True)
 return d

import argparse
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);aa=ap.parse_args();name=aa.case
if name=='default':case(name)
elif name.startswith('beta'):case(name,beta=float(name[4:]))
elif name.startswith('gamma'):case(name,gamma=float(name[5:]))
elif name.startswith('topk'):case(name,k=int(name[4:]))
elif name.startswith('metis'):case(name,seedrows=max(1,round(n*float(name[5:]))))
elif name.startswith('order'):case(name,order='reverse' if name=='orderreverse' else int(name[5:]))
elif name.startswith('batch'):case(name,batch=int(name[5:]))
