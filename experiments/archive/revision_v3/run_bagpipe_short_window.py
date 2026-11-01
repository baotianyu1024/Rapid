from pathlib import Path
import subprocess,os,concurrent.futures,json
V=Path(__file__).resolve().parents[1];P='/mnt/hugui/longjiu.bty/reproduction/torchrec_compare_env/bin/python';env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def run(ds):
 d=V/f'results/{ds}_8'
 with open(V/f'logs/{ds}_bagpipe_la8.log','w') as f:rc=subprocess.call([P,str(V/'code/bagpipe_count.py'),'--data',str(d/'data'),'--out',str(d/'bagpipe_la8'),'--lookahead','8'],stdout=f,stderr=subprocess.STDOUT,env=env)
 print(ds,rc,flush=True);return ds,rc
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:r=list(ex.map(run,['criteo','kaggle','taobao','t180','t200']))
(V/'results/bagpipe_la8_status.json').write_text(json.dumps(r,indent=2))
