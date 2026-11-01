from pathlib import Path
import time,subprocess,sys,os,json
V=Path(__file__).resolve().parents[1];p=V/'results/sensitivity_full/status.json'
while not p.exists():time.sleep(10)
assert all(rc==0 for name,rc in json.loads(p.read_text()))
with open(V/'logs/report_sensitivity.log','w') as f:r=subprocess.run([sys.executable,str(V/'code/report_sensitivity.py')],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
print(r.returncode,flush=True)
