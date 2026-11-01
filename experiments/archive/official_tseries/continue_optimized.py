"""Execute the unchanged selected recipe suffix; adjust only batch concurrency."""
import argparse,json,os,subprocess,sys
from pathlib import Path
O=Path(__file__).resolve().parents[1];V=O.parent;X=V/'code/reproduction/rapid_peak_optimization';p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--stage',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();case=O/f'results/{a.dataset}_8';out=case/'optimized';out.mkdir(exist_ok=True);P=sys.executable
cmd=[P,str(X/'run_optimized_kaggle.py'),'--ranks','8','--baseline',str(case/'baseline'),'--gmp-meta',str(case/'gmp_traffic/het_gmp_s0.json'),'--out',str(out),'--plan-only'];plan=json.loads(subprocess.check_output(cmd,text=True));start=next(i for i,(name,args) in enumerate(plan['steps']) if name==a.stage)
for name,cmd in plan['steps'][start:]:
 if '--workers' in cmd:cmd[cmd.index('--workers')+1]=str(a.workers)
 (out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2));print('START',name,flush=True)
 with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
 print('DONE',name,flush=True)
(out/'complete.json').write_text(json.dumps(dict(success=True,recipe='selected_peak_v1',ranks=8,final_case=plan['final_case'],cache_result=plan['cache_result'],original_global_batches_preserved=True,resumed_from=a.stage,batch_parallel_workers=a.workers,algorithm_parameters_unchanged=True),indent=2))
