"""RAPID symmetric peak balancing + strict-s0 cache, using a verified latest baseline.
Run 8 ranks first, then freeze the recipe for 64-rank validation. Does not overwrite the original result.
"""
import argparse,subprocess,json,hashlib,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ranks',type=int,choices=[8,64],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
g=a.ranks;latest=R/('herald_rapid_gmp/results/default8/latest_rapid' if g==8 else 'het_gmp_rapid/results/rank64_balance/latest_rapid');traffic=R/('herald_rapid_gmp/results/default8/fresh_baselines/traffic' if g==8 else 'het_gmp_rapid/results/rank64_balance/traffic')
a.out=a.out.resolve();a.out.mkdir(exist_ok=False);base=latest/'rapid_owner_base';cache=latest/'profile_cache'
for source,info in json.load(open(latest/'source_manifest.json')).items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==info['sha256'],'Baseline source changed; rerun the latest baseline first'
assert (base/'scheduled/dataset_0.bin').exists(),'Restore baseline datasets before refinement'
def run(name,args):
 cmd=[sys.executable]+list(map(str,args));(a.out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2))
 with open(a.out/(name+'.log'),'w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
owner=a.out/'owners';samples=a.out/'samples';mirrors=a.out/'cache'
run('owners',[R/'scripts/joint_price_owners.py','--case',base,'--cache',cache,'--resume-from',latest/'rapid_guard_p5','--output',owner,'--library',ROOT/'owner_kernel.so','--forward',128,'--backward',128,'--storage',60,'--penalty',5,'--budget',.2,'--passes',4,'--hot-threshold',16,'--hot-first'])
run('materialize',[R/'scripts/refine_balanced_owners.py','--case',base,'--cache',cache,'--output',owner,'--materialize-existing','--uncompressed-maps'])
run('samples',[ROOT/'refine_samples.py','--case',owner,'--out',samples,'--rounds',4,'--workers',8])
run('profile',[ROOT/'prepare_sample_cache.py',samples])
run('cache',[ROOT/'cache_extension.py','--case',samples,'--cache',samples/'profile_cache','--gmp-meta',traffic/'het_gmp_s0.json','--out',mirrors])
run('cache_verify',[ROOT/'verify_cache.py',mirrors])
files=[ROOT/'sample_peak.cpp',ROOT/'sample_peak.so',ROOT/'refine_samples.py',ROOT/'prepare_sample_cache.py',ROOT/'owner_kernel.cpp',ROOT/'owner_kernel.so',ROOT/'cache_extension.py',ROOT/'cache_masks.cpp',ROOT/'cache_masks.so',ROOT/'cache_oracle.cpp',ROOT/'cache_oracle.so',R/'scripts/joint_price_owners.py',R/'scripts/joint_price_owners.cpp',R/'scripts/joint_price_owners.so']
(a.out/'source_sha256.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2))
(a.out/'complete.json').write_text(json.dumps(dict(success=True,ranks=g,recipe='symmetric 128/128, storage60, penalty5, budget.2, 4passes, hot-first16; 4 rounds of capacity-preserving sample swaps; exact-benefit strict-s0 cache <= GMP mirror budgets'),indent=2))
