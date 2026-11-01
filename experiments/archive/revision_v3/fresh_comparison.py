"""Fresh revision evidence. Raw inputs only; no historical plans, profiles or counts."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
# T-series always uses the official generator pipeline; retired inspired files are never read.
_route=argparse.ArgumentParser(add_help=False);_route.add_argument('--dataset');_known,_unknown=_route.parse_known_args()
if _known.dataset in ['t180','t200','t510']:
 os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve().parents[1]/'official_tseries/code/run_comparison.py'),*sys.argv[1:]])

HERE=Path(__file__).resolve().parent;HOME=HERE.parent;WORKSPACE=Path("/mnt/hugui/longjiu.bty");R=HERE/'reproduction';X=R/'rapid_peak_optimization';H=R/'het_gmp_rapid/scripts';S=R/'scripts';W=R/'working';P=sys.executable
sys.path.insert(0,str(H));from common import measured,save
L=[3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1]
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['criteo','kaggle','taobao'],required=True);p.add_argument('--ranks',type=int,choices=[8,64],default=8);p.add_argument('--rows',type=int,default=1048576);a=p.parse_args();g=a.ranks;ds=a.dataset;b=8192;out=HOME/'results'/f'{ds}_{g}';out.mkdir(exist_ok=False);nb=a.rows//(b*g);assert nb*g*b==a.rows
lengths=[1]*4 if ds=='taobao' else L;dense=1 if ds=='taobao' else 13
src=WORKSPACE/('data/processed/taobao/taobao_4tables.bin' if ds=='taobao' else 'data/processed/kaggle/multihot.bin' if ds=='kaggle' else 'data/processed/criteo/day6.bin')
data=out/'data';baseout=out/'baseline';baseout.mkdir();model=baseout/'rapid_schedule';model.mkdir()
os.environ['LD_LIBRARY_PATH']=str(WORKSPACE/'reproduction/toolchain/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
def run(name,args):return measured(name,args,out,threads=8)
manifest={str(f):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for d in [S,X,H,W/'cps'] for f in d.iterdir() if f.is_file() and (f.suffix in ['.py','.cpp','.h','.so','.sh'] or f.name in ['arrange_ids_to_devices','generate_local_dataset'])};save(baseout/'source_manifest.json',manifest)
save(out/'experiment.json',dict(dataset=ds,rows=a.rows,ranks=g,local_batch=b,batches=nb,epochs=5,dim=128,bytes_per_component=4,source=str(src),cpu_logical_replay=True,fresh=True,recipe='current offline refined RAPID; optional strict-s0 cache disclosed separately',start=time.time()))
run('prepare',[P,H/'prepare.py','--input',src,'--out',data,'--batches',nb,'--ranks',g,'--batch-size',b,'--dataset',ds,'--dense',dense,'--table-lengths',*lengths])
run('gmp_partition',[P,H/'partition_het.py','--data',data,'--out',out/'gmp','--ranks',g,'--batch-size',b])
run('gmp_replay',[P,H/'replay_het.py','--data',data,'--partition',out/'gmp','--out',out/'gmp_traffic','--ranks',g,'--batch-size',b,'--epochs',5,'--bounds',0,'--skip-1d'])
cfg=dict(dataset=ds,input=str(data/'input.bin'),output=str(model),repo=str(W),ranks=g,batches=nb,rows=a.rows,batch_size=b,dense=dense,table_lengths=lengths,topk=40000,hot_rows=a.rows,base_case=None,gpu_execution=False);save(model/'config.json',cfg)
common=['--dense_count',dense,'--table_lengths',*lengths]
run('rapid_hot',[P,W/'cps/hot_id_extraction.py','--input',data/'input.bin','--output',model/'hot','--topk',40000,'--max_rows',a.rows,*common])
run('rapid_placement',[W/'cps/arrange_ids_to_devices','--input_bin',data/'input.bin','--hot_ids_dir',model/'hot','--output_dir',model/'placement','--batch_size',b,'--num_devices',g,'--n_batches',nb,*common])
run('rapid_schedule',[W/'cps/generate_local_dataset','--input_file',data/'input.bin','--table_files_dir',model/'placement','--output_dir',model/'scheduled','--num_devices',g,'--train_batch_size',b,'--num_chunks',nb,'--steps_per_chunk',1,'--num_threads',8,'--gamma',.2,'--swap_algorithm','ratio','--swap_screen_queries_per_gpu',20000000,'--swap_evaluations_per_gpu',500000,*common])
base=baseout/'rapid_owner_base';run('owner_base',[P,S/'refine_trace_owners.py','--case',model,'--output',base,'--target-rank',7,'--target-fraction',.1,'--no-base-counts','--uncompressed-maps'])
previous=None
for suffix,f,b,s,pen,budget,passes,hot in [('price_f2',2,1,30,.2,.05,4,0),('price_b8',4,8,30,.5,.05,3,32),('price_b32',8,32,60,.5,.05,3,16),('guard_p5',8,32,60,5,.05,3,16)]:
 dest=baseout/('rapid_'+suffix);cmd=[P,S/'joint_price_owners.py','--case',base,'--cache',baseout/'profile_cache','--output',dest,'--forward',f,'--backward',b,'--storage',s,'--penalty',pen,'--budget',budget,'--passes',passes,'--hot-threshold',hot]
 cmd+=['--allow-reduction-tables']
 if previous:cmd+=['--resume-from',previous]
 if suffix=='guard_p5':cmd+=['--guard-case',previous]
 # Unique optimization is valid here only when all tables pass its explicit guard.
 run(suffix,cmd);previous=dest
run('optimized',[P,X/'run_optimized_kaggle.py','--ranks',g,'--baseline',baseout,'--gmp-meta',out/'gmp_traffic/het_gmp_s0.json','--out',out/'optimized'])
save(out/'complete.json',dict(success=True,finished=time.time()))
