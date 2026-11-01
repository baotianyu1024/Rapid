"""Improve only cases whose CPS receive peak exceeds the paper reference."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reproduction';refs=json.loads((R/'paper_reference_targets.json').read_text())
jobs=[(f'criteo_day6_{g}_updated_{b}b','ratio') for g,b in [(8,64),(16,32),(32,16),(64,8),(128,4)]]
jobs += [('kaggle_8_updated_64b','weighted'),('kaggle_16_updated_32b','weighted'),('taobao_64_updated_20b','weighted')]
def improve(pair):
 name,algorithm=pair;case=R/'results'/name
 while not (case/'complete.json').exists():time.sleep(15)
 cfg=json.loads((case/'config.json').read_text());d=json.loads((case/'cps_metrics.json').read_text());ref=refs[cfg['dataset']][str(cfg['ranks'])]['cps']
 if d['acp_receive']['mean_batch_peak']<=ref:
  print('already better; no changes',name,d['acp_receive']['mean_batch_peak'],ref,flush=True);return
 out=R/'results'/(name+'_improved')
 cmd=[sys.executable,str(R/'scripts/run_variant.py'),'--case',str(case),'--output',str(out),'--algorithm',algorithm,'--evaluation-budget','500000']
 if algorithm=='ratio':cmd+=['--screen-budget','20000000']
 else:cmd+=['--weight','16']
 with (R/'logs'/(out.name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 improved=json.loads((out/'cps_metrics.json').read_text())['acp_receive']['mean_batch_peak']
 print(name,'paper',ref,'before',d['acp_receive']['mean_batch_peak'],'after',improved,flush=True)
jobs=jobs[-3:]+jobs[:-3]
with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(improve,jobs))
