from pathlib import Path
import argparse,os,sys,json,hashlib
V=Path(__file__).resolve().parents[1];R=V/'code/reproduction';S=R/'scripts';X=R/'rapid_peak_optimization';H=R/'het_gmp_rapid/scripts';W=R/'working';P=sys.executable;sys.path.insert(0,str(H));from common import measured,save
p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();out=V/'results/sensitivity_full'/a.case;out.mkdir(parents=True,exist_ok=False);baseout=out/'baseline';baseout.mkdir();model=V/'results/sensitivity'/a.case;g=8
(baseout/'rapid_schedule').symlink_to(model.resolve(),target_is_directory=True)
manifest={str(f):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for d in [S,X,H,W/'cps'] for f in d.iterdir() if f.is_file() and (f.suffix in ['.py','.cpp','.h','.so','.sh'] or f.name in ['arrange_ids_to_devices','generate_local_dataset'])};save(baseout/'source_manifest.json',manifest)
def run(name,args):return measured(name,args,out,threads=4)
base=baseout/'rapid_owner_base';run('owner_base',[P,S/'refine_trace_owners.py','--case',model,'--output',base,'--target-rank',7,'--target-fraction',.1,'--no-base-counts','--uncompressed-maps'])
previous=None
for suffix,f,b,s,pen,budget,passes,hot in [('price_f2',2,1,30,.2,.05,4,0),('price_b8',4,8,30,.5,.05,3,32),('price_b32',8,32,60,.5,.05,3,16),('guard_p5',8,32,60,5,.05,3,16)]:
 dest=baseout/('rapid_'+suffix);cmd=[P,S/'joint_price_owners.py','--case',base,'--cache',baseout/'profile_cache','--output',dest,'--forward',f,'--backward',b,'--storage',s,'--penalty',pen,'--budget',budget,'--passes',passes,'--hot-threshold',hot]
 cmd+=['--allow-reduction-tables']
 if previous:cmd+=['--resume-from',previous]
 if suffix=='guard_p5':cmd+=['--guard-case',previous]
 # Unique optimization is valid here only when all tables pass its explicit guard.
 run(suffix,cmd);previous=dest
run('optimized',[P,X/'run_optimized_kaggle.py','--ranks',g,'--baseline',baseout,'--out',out/'optimized'])
