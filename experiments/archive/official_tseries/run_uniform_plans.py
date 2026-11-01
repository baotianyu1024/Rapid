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
 case=O/f'results/{ds}_8'
 call(ds,'torchrec_uniform128',[TP,O/'code/torchrec_plan.py','--data',case/'data','--out',case/'torchrec_uniform128/plan.json','--hbm-gib',16,'--intra-gib-s',14.66833055,'--uniform-dim',128])
 return ds
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for ds in ex.map(run,['t180','t200']):print('COMPLETE',ds,flush=True)
