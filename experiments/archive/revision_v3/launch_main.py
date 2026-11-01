from pathlib import Path
import subprocess,sys,os,json,time
V=Path(__file__).resolve().parents[1];env=dict(os.environ,LD_LIBRARY_PATH='/mnt/hugui/longjiu.bty/reproduction/toolchain/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8');jobs=[]
for ds in ['criteo','kaggle','taobao']:
 f=open(V/f'logs/{ds}_8.log','w');p=subprocess.Popen([sys.executable,str(V/'code/fresh_comparison.py'),'--dataset',ds,'--ranks','8'],stdout=f,stderr=subprocess.STDOUT,env=env);jobs.append((ds,p,f));print(ds,p.pid,flush=True)
for ds,p,f in jobs:
 rc=p.wait();f.close();print('DONE',ds,rc,flush=True)
