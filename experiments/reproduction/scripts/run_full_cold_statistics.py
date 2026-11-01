from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parents[2]
while not (root/'data/processed/criteo/day6.json').exists():time.sleep(15)
subprocess.run([sys.executable,str(root/'reproduction/scripts/cold_statistics.py'),'--input',str(root/'data/processed/criteo/day6.bin'),'--rows','0','--mcr-sample','512','--output',str(root/'reproduction/results/criteo_day6_cold_statistics')],check=True)
