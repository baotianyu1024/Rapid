from pathlib import Path
import subprocess,os,sys,concurrent.futures,json
V=Path(__file__).resolve().parents[1];env=dict(os.environ,LD_LIBRARY_PATH='/mnt/hugui/longjiu.bty/reproduction/toolchain/lib',PYTHONPATH=str(V/'dependencies')+':/mnt/hugui/longjiu.bty/reproduction/deps',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
def one(name):
 with open(V/f'logs/retry_{name}.log','w') as f:
  for script in ['sensitivity_initial.py','sensitivity_finish.py']:
   r=subprocess.run([sys.executable,str(V/'code'/script),'--case',name],stdout=f,stderr=subprocess.STDOUT,env=env)
   if r.returncode:break
 print(name,r.returncode,flush=True);return name,r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:r=list(ex.map(one,['metis1e-05','metis0.0001','metis0.001']))
(V/'results/sensitivity_full/metis_retry_status.json').write_text(json.dumps(r,indent=2))
