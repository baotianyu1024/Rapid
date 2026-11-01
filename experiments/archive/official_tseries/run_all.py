import concurrent.futures,json,os,subprocess,time
from pathlib import Path
O=Path(__file__).resolve().parents[1];P='/home/longjiu.bty/miniconda3/envs/het-gmp-rapid/bin/python';TP='/mnt/hugui/longjiu.bty/reproduction/torchrec_compare_env/bin/python'
env=dict(os.environ,LD_LIBRARY_PATH='/mnt/hugui/longjiu.bty/reproduction/toolchain/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',PYTHONPATH=str(O.parent/'dependencies')+':/mnt/hugui/longjiu.bty/reproduction/deps')
def call(ds,tag,cmd):
 with (O/f'logs/{ds}_{tag}.log').open('w') as f:
  rc=subprocess.call(list(map(str,cmd)),env=env,stdout=f,stderr=subprocess.STDOUT)
 (O/f'logs/{ds}_{tag}.status.json').write_text(json.dumps(dict(returncode=rc,command=list(map(str,cmd)),time=time.time())))
 if rc:raise RuntimeError((ds,tag,rc))
def run(ds):
 while True:
  m=json.loads((O/f'datasets/{ds}.json').read_text())
  if m.get('complete'):break
  time.sleep(10)
 case=O/f'results/{ds}_8';data=case/'data'
 call(ds,'comparison',[P,O/'code/run_comparison.py','--dataset',ds,'--sample-workers',16 if ds=='t180' else 8]+(['--resume'] if case.exists() else []))
 jobs=[('gmp_bytes',[P,O/'code/replay_het_bytes.py','--data',data,'--partition',case/'gmp','--out',case/'gmp_bytes','--ranks',8,'--batch-size',512,'--epochs',5,'--bounds',0,'--skip-1d']),('herald',[P,O/'herald/scripts/run_herald.py','--data',data,'--out',case/'herald']),('bagpipe',[TP,O/'code/bagpipe_count.py','--data',data,'--out',case/'bagpipe_la8','--lookahead',8]),('torchrec',[TP,O/'code/torchrec_plan.py','--data',data,'--out',case/'torchrec/plan.json','--hbm-gib',16,'--intra-gib-s',14.66833055])]
 (case/'torchrec').mkdir(exist_ok=True);(case/'torchrec_uniform128').mkdir(exist_ok=True)
 jobs.append(('torchrec_uniform128',[TP,O/'code/torchrec_plan.py','--data',data,'--out',case/'torchrec_uniform128/plan.json','--hbm-gib',16,'--intra-gib-s',14.66833055,'--uniform-dim',128]))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  futures=[ex.submit(call,ds,tag,cmd) for tag,cmd in jobs]
  for f in futures:f.result()
 return ds
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 futures=[ex.submit(run,ds) for ds in ['t180','t200','t510']]
 for f in concurrent.futures.as_completed(futures):print('COMPLETE',f.result(),flush=True)
