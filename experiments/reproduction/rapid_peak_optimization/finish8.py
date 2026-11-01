import subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
for name,args in [('r8_samples_cache',[ROOT/'cache_extension.py','--case',ROOT/'r8_samples','--cache',ROOT/'r8_samples/profile_cache','--gmp-meta',R/'herald_rapid_gmp/results/default8/fresh_baselines/traffic/het_gmp_s0.json','--out',ROOT/'r8_samples_cache']),('r8_verify_cache',[ROOT/'verify_cache.py',ROOT/'r8_samples_cache'])]:
 with open(ROOT/(name+'.log'),'w') as f:subprocess.run([sys.executable]+list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
 print('DONE',name,flush=True)
