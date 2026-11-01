from pathlib import Path
import subprocess,os,json,concurrent.futures
V=Path(__file__).resolve().parents[1];P='/home/longjiu.bty/miniconda3/envs/het-gmp-rapid/bin/python';TP='/mnt/hugui/longjiu.bty/reproduction/torchrec_compare_env/bin/python';env=dict(os.environ,LD_LIBRARY_PATH='/mnt/hugui/longjiu.bty/reproduction/toolchain/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
def run(ds,method):
 case=V/f'results/{ds}_8';out=case/method;out.mkdir(exist_ok=True);data=case/'data'
 if method=='herald':cmd=[P,str(V/'code/reproduction/herald_rapid_gmp/scripts/run_herald.py'),'--data',str(data),'--out',str(out)]
 elif method=='bagpipe':cmd=[TP,str(V/'code/bagpipe_count.py'),'--data',str(data),'--out',str(case/'bagpipe_la8'),'--lookahead','8']
 else:cmd=[TP,str(V/'code/torchrec_plan.py'),'--data',str(data),'--out',str(out/'plan.json'),'--dim','128','--hbm-gib','16','--intra-gib-s','14.66833055']
 with open(V/f'logs/{ds}_{method}.log','w') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 print(ds,method,rc,flush=True);return ds,method,rc
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
 fs=[ex.submit(run,ds,m) for ds in ['criteo','kaggle','taobao'] for m in ['torchrec','herald','bagpipe']]
 results=[f.result() for f in fs]
(V/'results/baseline_status.json').write_text(json.dumps(results,indent=2))
