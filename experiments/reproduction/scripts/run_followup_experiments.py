"""Launch dependent ablations only after their baseline is complete."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
def ablate(pair):
 case,name=pair;base=ROOT/'reproduction/results'/case;out=ROOT/'reproduction/results'/name
 while not (base/'complete.json').exists():time.sleep(15)
 if (out/'metrics.json').exists() and (out/'config.json').exists():return
 with (ROOT/'reproduction/logs'/f'{name}.log').open('w') as f:
  subprocess.run([sys.executable,str(ROOT/'reproduction/scripts/run_ablation.py'),'--case',str(base),'--output',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('complete',name,flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 list(pool.map(ablate,[('kaggle_32_updated_16b','kaggle_32_ablation_16b'),('criteo_day6_32_updated_16b','criteo_day6_32_ablation_16b')]))
