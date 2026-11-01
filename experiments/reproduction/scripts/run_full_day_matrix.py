"""Run a common 4,194,304-row trace after the full-day shuffle is committed."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
meta=ROOT/'data/processed/criteo/day6.json'
while not meta.exists():time.sleep(15)
assert json.loads(meta.read_text())['rows']==204846845
jobs=[(8,64),(16,32),(32,16),(64,8),(128,4)]
def run(job):
 g,b=job;name=f'criteo_day6_{g}_updated_{b}b';out=ROOT/'reproduction/results'/name
 if (out/'complete.json').exists():return
 cmd=[sys.executable,str(ROOT/'reproduction/scripts/run_case.py'),'--input',str(meta.with_suffix('.bin')),'--dataset','criteo','--repo',str(ROOT/'reproduction/baseline/update_Rapid'),'--ranks',str(g),'--batches',str(b),'--output',str(out)]
 with (ROOT/'reproduction/logs'/f'{name}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('complete',name,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
