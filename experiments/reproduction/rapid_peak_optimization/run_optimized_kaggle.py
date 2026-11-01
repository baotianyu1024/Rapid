"""Default to the selected, verified RAPID peak/balance recipe (8/64-rank trace preset).
All sample transformations preserve original global batches. Strict-s0 mirrors
are an explicit extension with the same per-rank capacity upper bound as GMP.
"""
import argparse,subprocess,json,hashlib,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ranks',type=int,choices=[8,64],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--plan-only',action='store_true');p.add_argument('--baseline',type=Path,help='Fresh dataset-specific baseline directory');p.add_argument('--gmp-meta',type=Path,help='Matched GMP replay metadata');a=p.parse_args();g=a.ranks
latest=R/('herald_rapid_gmp/results/default8/latest_rapid' if g==8 else 'het_gmp_rapid/results/rank64_balance/latest_rapid');traffic=R/('herald_rapid_gmp/results/default8/fresh_baselines/traffic' if g==8 else 'het_gmp_rapid/results/rank64_balance/traffic');latest=a.baseline.resolve() if a.baseline else latest;base=latest/'rapid_owner_base';cache=latest/'profile_cache';a.out=a.out.resolve()
cfg=json.load(open(base/'config.json'));assert cfg['ranks']==g and 1<=cfg['batches']<=64
for source,info in json.load(open(latest/'source_manifest.json')).items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==info['sha256'],'Baseline source changed; rerun the latest baseline first'
steps=[]
def add(name,args):steps.append((name,[sys.executable]+list(map(str,args))))
def profile(case):add(case.name+'_profile',[ROOT/'prepare_sample_cache.py',case])
def materialize(source,cache,case):add(case.name+'_materialize',[R/'scripts/refine_balanced_owners.py','--case',source,'--cache',cache,'--output',case,'--materialize-existing','--uncompressed-maps'])
owner=a.out/'symmetric';s4=a.out/'samples4';cold=a.out/'cold_repair';s16=a.out/'samples16';soft=a.out/'batch_soft';guard=a.out/'hot_guard';budget=a.out/'hot_budget';alternate=a.out/'alternate2';peak=a.out/'peak_samples';mirrors=a.out/'cache'
args=[R/'scripts/joint_price_owners.py','--case',base,'--cache',cache,'--resume-from',latest/'rapid_guard_p5','--output',owner,'--forward',128,'--backward',128,'--storage',60,'--penalty',5,'--budget',.2,'--passes',4,'--hot-threshold',16,'--hot-first','--allow-reduction-tables']
if g==64:args+=['--library',ROOT/'owner_kernel.so']
add('symmetric',args);materialize(base,cache,owner)
add('samples4',[ROOT/'refine_samples.py','--case',owner,'--out',s4,'--rounds',4,'--workers',8]);profile(s4)
add('cold_repair',[ROOT/'cold_capacity.py','--case',s4,'--cache',s4/'profile_cache','--out',cold]);materialize(s4,s4/'profile_cache',cold)
add('samples16',[ROOT/'refine_samples.py','--case',cold,'--out',s16,'--rounds',16,'--workers',8]);profile(s16)
add('batch_soft',[ROOT/'batch_soft_samples.py','--case',s16,'--out',soft,'--rounds',4,'--balance',1]);profile(soft)
for source,dest,script in [(soft,guard,'swap_hot_guarded.py'),(guard,budget,'swap_hot_budget.py')]:
 add(dest.name,[ROOT/script,'--case',source,'--cache',source/'profile_cache','--out',dest,'--passes',2]);materialize(source,source/'profile_cache',dest);profile(dest)
for source,dest,script in [(budget,alternate,'batch_soft_samples.py'),(alternate,peak,'batch_peak_samples.py')]:
 add(dest.name,[ROOT/script,'--case',source,'--out',dest,'--rounds',4,'--balance',4]);profile(dest)
if a.plan_only:print(json.dumps(dict(recipe='selected_peak_v1',steps=steps,final_case=str(peak),cache_result=str(mirrors)),indent=2));sys.exit(0)
a.out.mkdir(exist_ok=False)
for name,cmd in steps:
 (a.out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2));print('START',name,flush=True)
 with open(a.out/(name+'.log'),'w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
 print('DONE',name,flush=True)
(a.out/'complete.json').write_text(json.dumps(dict(success=True,recipe='selected_peak_v1',ranks=g,final_case=str(peak),cache_result=str(mirrors),original_global_batches_preserved=True),indent=2))
