from pathlib import Path
import time,subprocess,sys,os,json
V=Path(__file__).resolve().parents[1];names=['default','beta0','beta0.001','gamma0','gamma1','topk10000','topk100000','metis1e-05','metis0.0001','metis0.001','orderreverse','order11','order29','batch512','batch1024','batch2048','batch4096','batch16384']
while not all((V/f'results/sensitivity_full/{n}/optimized/complete.json').exists() for n in names):time.sleep(10)
with open(V/'logs/report_sensitivity.log','w') as f:r=subprocess.run([sys.executable,str(V/'code/report_sensitivity.py')],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
(V/'results/sensitivity_full/resolved_status.json').write_text(json.dumps(dict(success=r.returncode==0,cases=names,metis_retry='Missing typing_extensions in isolated copy fixed by copying dependency; initial failure logs retained'),indent=2));print(r.returncode,flush=True)
