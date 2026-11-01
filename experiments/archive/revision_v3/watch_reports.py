from pathlib import Path
import time,subprocess,sys,os
V=Path(__file__).resolve().parents[1];names=['taobao_8','criteo_8','kaggle_8','t180_8','t200_8','criteo_64'];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4');pending=set(names)
while pending:
 for n in list(pending):
  d=V/'results'/n;ready=(d/'complete.json').exists()
  if n!='criteo_64':ready=ready and all((d/z).exists() for z in ['herald/result.json','bagpipe/result.json','torchrec/plan.json'])
  if ready:
   with open(V/f'logs/report_{n}.log','w') as f:rc=subprocess.call([sys.executable,str(V/'code/report_comparisons.py'),'--case',n],stdout=f,stderr=subprocess.STDOUT,env=env)
   print(n,rc,flush=True);pending.remove(n)
 if pending:time.sleep(10)
