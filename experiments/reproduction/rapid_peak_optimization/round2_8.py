import subprocess,sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
steps=[('r8_cold_materialize',[R/'scripts/refine_balanced_owners.py','--case',ROOT/'r8_samples','--cache',ROOT/'r8_samples/profile_cache','--output',ROOT/'r8_cold_repair','--materialize-existing','--uncompressed-maps']),('r8_capacity_samples',[ROOT/'refine_samples.py','--case',ROOT/'r8_cold_repair','--out',ROOT/'r8_capacity_samples','--rounds',16,'--workers',8]),('r8_capacity_profile',[ROOT/'prepare_sample_cache.py',ROOT/'r8_capacity_samples']),('r8_capacity_samples_cache',[ROOT/'cache_extension_v2.py','--case',ROOT/'r8_capacity_samples','--cache',ROOT/'r8_capacity_samples/profile_cache','--gmp-meta',R/'herald_rapid_gmp/results/default8/fresh_baselines/traffic/het_gmp_s0.json','--out',ROOT/'r8_capacity_samples_cache']),('r8_capacity_verify_cache',[ROOT/'verify_cache.py',ROOT/'r8_capacity_samples_cache'])]
for name,args in steps:
 cmd=[sys.executable]+list(map(str,args));(ROOT/(name+'.command.json')).write_text(json.dumps(cmd,indent=2));print('START',name,flush=True)
 with open(ROOT/(name+'.log'),'w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
 print('DONE',name,flush=True)
