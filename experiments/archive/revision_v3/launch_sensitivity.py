from pathlib import Path
import subprocess,os,sys,concurrent.futures,json
V=Path(__file__).resolve().parents[1];env=dict(os.environ,LD_LIBRARY_PATH='/mnt/hugui/longjiu.bty/reproduction/toolchain/lib',PYTHONPATH='/mnt/hugui/longjiu.bty/reproduction/revision_response_20261026/dependencies:/mnt/hugui/longjiu.bty/reproduction/deps',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
def one(name):
 with open(V/f'logs/sensitivity_{name}.log','w') as f:
  for script in ['sensitivity_initial.py','sensitivity_finish.py']:
   rc=subprocess.call([sys.executable,str(V/'code'/script),'--case',name],stdout=f,stderr=subprocess.STDOUT,env=env)
   if rc:break
 print(name,rc,flush=True);return name,rc
# Prepare shared hot sets sequentially to avoid writers racing.
for name in ['default','topk10000','topk100000']:one(name)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 fs=[ex.submit(one,n) for n in ['beta0','beta0.001','gamma0','gamma1','metis1e-05','metis0.0001','metis0.001','orderreverse','order11','order29','batch512','batch1024','batch2048','batch4096','batch16384']]
 r=[f.result() for f in fs]
(V/'results/sensitivity_full/status.json').write_text(json.dumps(r,indent=2))
