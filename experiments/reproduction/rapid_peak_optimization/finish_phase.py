import subprocess,sys,time,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT.parent

def run(name,args):
 cmd=[sys.executable]+list(map(str,args));(ROOT/(name+'.command.json')).write_text(json.dumps(cmd,indent=2));print('START',name,flush=True)
 with open(ROOT/(name+'.log'),'w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
 print('DONE',name,flush=True)
while not (ROOT/'r8_guard_phase/refinement.json').exists():time.sleep(5)
for g in [8,64]:
 base=ROOT/f'r{g}_guard_budget';case=ROOT/f'r{g}_guard_phase';cache=ROOT/f'r{g}_guard_phase_cache'
 if g==64:
  while not (base/'storage.json').exists():time.sleep(5)
  run('r64_guard_phase',[ROOT/'swap_hot_phase.py','--case',base,'--cache',base/'profile_cache','--out',case,'--passes',2])
 run(f'r{g}_guard_phase_materialize',[R/'scripts/refine_balanced_owners.py','--case',base,'--cache',base/'profile_cache','--output',case,'--materialize-existing','--uncompressed-maps'])
 run(f'r{g}_guard_phase_profile',[ROOT/'prepare_sample_cache.py',case])
 meta=R/'herald_rapid_gmp/results/default8/fresh_baselines/traffic/het_gmp_s0.json' if g==8 else R/'het_gmp_rapid/results/rank64_balance/traffic/het_gmp_s0.json'
 run(cache.name,[ROOT/'cache_extension_v2.py','--case',case,'--cache',case/'profile_cache','--gmp-meta',meta,'--out',cache])
 run(f'r{g}_guard_phase_oracle',[ROOT/'verify_cache.py',cache])
 m=json.load(open(cache/'result.json'));print('RESULT',g,m['metrics']['combined_receive']['mean_step_peak'],m['phase_bottleneck_mean'],flush=True)
 if g==8 and (m['metrics']['combined_receive']['mean_step_peak']>=194602.4127 or m['phase_bottleneck_mean']>=197737.1238):print('STOP: failed 8-rank GMP criterion',flush=True);break
