import subprocess,os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT.parent;latest=R/'het_gmp_rapid/results/rank64_balance/latest_rapid';base=latest/'rapid_owner_base';cache=latest/'profile_cache';owner=ROOT/'r64_symmetric';samples=ROOT/'r64_samples';mirrors=ROOT/'r64_samples_cache'
selected=json.load(open(ROOT/'comparison.json'))['8'];assert selected['r8_samples']['combined_receive']['full_mean_peak']<selected['GMP']['combined_receive']['full_mean_peak']
recipe=dict(forward=128,backward=128,storage=60,penalty=5,budget=.2,passes=4,hot_first=True,hot_threshold=16,sample_rounds=4,sample_sources='every rank',sample_targets=4,screen_samples=1024,candidates=[64,32],cache='strict-s0; exact avoided-fetch benefit; at most GMP per-rank mirror budgets')
(ROOT/'frozen_recipe.json').write_text(json.dumps(dict(selected_on_ranks=8,validation_ranks=64,recipe=recipe,evidence=selected['r8_samples'],sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'owner_kernel.cpp',ROOT/'owner_kernel.so',ROOT/'sample_peak.cpp',ROOT/'sample_peak.so',ROOT/'refine_samples.py',ROOT/'cache_extension.py',ROOT/'cache_masks.cpp',ROOT/'cache_masks.so']}),indent=2))
def run(name,args):
 cmd=[sys.executable]+list(map(str,args));(ROOT/(name+'.command.json')).write_text(json.dumps(cmd,indent=2));print('START',name,flush=True)
 with open(ROOT/(name+'.log'),'w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
 print('DONE',name,flush=True)
run('r64_owner',[R/'scripts/joint_price_owners.py','--case',base,'--cache',cache,'--resume-from',latest/'rapid_guard_p5','--output',owner,'--library',ROOT/'owner_kernel.so','--forward',128,'--backward',128,'--storage',60,'--penalty',5,'--budget',.2,'--passes',4,'--hot-threshold',16,'--hot-first'])
run('r64_materialize',[R/'scripts/refine_balanced_owners.py','--case',base,'--cache',cache,'--output',owner,'--materialize-existing','--uncompressed-maps'])
run('r64_samples',[ROOT/'refine_samples.py','--case',owner,'--out',samples,'--rounds',4,'--workers',8])
run('r64_profile',[ROOT/'prepare_sample_cache.py',samples])
run('r64_samples_cache',[ROOT/'cache_extension.py','--case',samples,'--cache',samples/'profile_cache','--gmp-meta',R/'het_gmp_rapid/results/rank64_balance/traffic/het_gmp_s0.json','--out',mirrors])
run('r64_verify_cache',[ROOT/'verify_cache.py',mirrors])
run('report',[ROOT/'report.py'])
(ROOT/'validation64_complete.json').write_text(json.dumps(dict(success=True,recipe='frozen_recipe.json'),indent=2))
