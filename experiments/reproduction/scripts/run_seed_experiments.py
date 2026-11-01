"""Run seed-only contrasts on identical full-day shuffled traces."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[2]
def run(g):
 b=512//g;name=f'criteo_day6_{g}_metis_seed_{b}b';base=root/f'reproduction/results/criteo_day6_{g}_updated_{b}b'
 cmd=[sys.executable,str(root/'reproduction/scripts/run_case.py'),'--input',str(root/'data/processed/criteo/day6.bin'),'--dataset','criteo','--repo',str(root/'reproduction/working'),'--ranks',str(g),'--batches',str(b),'--output',str(root/'reproduction/results'/name),'--seed-assignment',str(root/f'reproduction/metis_seed_day6/seed_{g}.txt'),'--base-case',str(base)]
 subprocess.run(cmd,check=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,[8,32,128]))
