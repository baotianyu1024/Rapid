import subprocess,sys,time,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT.parent

def run(name,args):
 cmd=[sys.executable]+list(map(str,args));(ROOT/(name+'.command.json')).write_text(json.dumps(cmd,indent=2));print('START',name,flush=True)
 with open(ROOT/(name+'.log'),'w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
 print('DONE',name,flush=True)
while not (ROOT/'r8_samples_v2/complete.json').exists():time.sleep(5)
for g in [8,64]:
 case=ROOT/f'r{g}_samples_v2';cache=ROOT/f'r{g}_samples_v2_cache'
 if g==64:run('r64_samples_v2',[ROOT/'refine_samples_v2.py','--case',ROOT/'r64_capacity_samples','--out',case,'--rounds',16,'--workers',8])
 run(f'r{g}_samples_v2_profile',[ROOT/'prepare_sample_cache.py',case])
 meta=R/'herald_rapid_gmp/results/default8/fresh_baselines/traffic/het_gmp_s0.json' if g==8 else R/'het_gmp_rapid/results/rank64_balance/traffic/het_gmp_s0.json'
 run(cache.name,[ROOT/'cache_extension_v2.py','--case',case,'--cache',case/'profile_cache','--gmp-meta',meta,'--out',cache])
 run(f'r{g}_samples_v2_oracle',[ROOT/'verify_cache.py',cache])
 m=json.load(open(cache/'result.json'));print('RESULT',g,m['metrics']['combined_receive']['mean_step_peak'],m['phase_bottleneck_mean'],flush=True)
 if g==8 and (m['metrics']['combined_receive']['mean_step_peak']>=188509.784375 or m['phase_bottleneck_mean']>=199842.68125):print('STOP: no joint 8-rank improvement',flush=True);break
