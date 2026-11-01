"""Fresh Taobao four-table comparison, fixed latest offline recipe, no incremental work."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
X=Path(__file__).resolve().parent;R=X.parent;H=R/'het_gmp_rapid/scripts';S=R/'scripts';W=R/'working';P=sys.executable
sys.path.insert(0,str(H))
from common import measured,save
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,choices=[8,64],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();g=a.ranks;out=a.out.resolve();out.mkdir(exist_ok=False)
nb=4194304//(8192*g);data=out/'data';baseout=out/'baseline';baseout.mkdir();model=baseout/'rapid_schedule';model.mkdir();lengths=[1]*4
os.environ['LD_LIBRARY_PATH']=str(R/'toolchain/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
def run(name,args):return measured(name,args,out,threads=16)
manifest={str(f):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for d in [S,X,H,W/'cps'] for f in d.iterdir() if f.is_file() and (f.suffix in ['.py','.cpp','.h','.so','.sh'] or f.name in ['arrange_ids_to_devices','generate_local_dataset'])}
save(baseout/'source_manifest.json',manifest)
save(out/'experiment.json',dict(dataset='taobao_4tables',rows=4194304,ranks=g,local_batch=8192,batches=nb,epochs=5,dim=128,bytes_per_component=4,recipe='run_optimized.py unchanged optimization parameters; table-count generalized',cpu_logical_replay=True,not_full_dataset=True,sample_constraint='original global batch preserved',mirrors='RAPID <= GMP per-rank budgets, strict s=0',source=str(R.parent/'data/processed/taobao/taobao_4tables.bin')))
run('prepare',[P,H/'prepare.py','--input',R.parent/'data/processed/taobao/taobao_4tables.bin','--out',data,'--batches',nb,'--ranks',g,'--dataset','taobao_4tables','--dense',1,'--table-lengths',*lengths])
run('gmp_partition',[P,H/'partition_het.py','--data',data,'--out',out/'gmp','--ranks',g])
run('gmp_replay',[P,H/'replay_het.py','--data',data,'--partition',out/'gmp','--out',out/'gmp_traffic','--ranks',g,'--epochs',5,'--bounds',0,'--skip-1d'])
run('gmp_batch_constraint',[P,X/'constrain_gmp.py','--ranks',g,'--source',out/'gmp','--data',data,'--out',out/'gmp_fixed'])
cfg=dict(dataset='taobao',input=str(data/'input.bin'),output=str(model),repo=str(W),ranks=g,batches=nb,rows=4194304,batch_size=8192,dense=1,table_lengths=lengths,topk=40000,hot_rows=4194304,base_case=None,gpu_execution=False)
save(model/'config.json',cfg)
old=R/'het_gmp_rapid/results/timing8_rerun'
def replace(cmd,values):
 cmd=list(cmd)
 i=cmd.index('--table_lengths')+1;j=i
 while j<len(cmd) and not cmd[j].startswith('--'):j+=1
 cmd[i:j]=list(map(str,lengths));values['--dense_count']=1
 for k,v in values.items():cmd[cmd.index(k)+1]=str(v)
 return cmd
cmd=replace(json.loads((old/'rapid_hot.command.json').read_text()),{'--input':data/'input.bin','--output':model/'hot'});cmd[:2]=[P,str(W/'cps/hot_id_extraction.py')];run('rapid_hot',cmd)
cmd=replace(json.loads((old/'rapid_placement.command.json').read_text()),{'--input_bin':data/'input.bin','--hot_ids_dir':model/'hot','--output_dir':model/'placement','--num_devices':g,'--n_batches':nb});cmd[0]=str(W/'cps/arrange_ids_to_devices');run('rapid_placement',cmd)
cmd=replace(json.loads((old/'rapid_schedule.command.json').read_text()),{'--input_file':data/'input.bin','--table_files_dir':model/'placement','--output_dir':model/'scheduled','--num_devices':g,'--num_chunks':nb,'--num_threads':16});cmd[0]=str(W/'cps/generate_local_dataset');run('rapid_schedule',cmd)
base=baseout/'rapid_owner_base'
run('owner_base',[P,S/'refine_trace_owners.py','--case',model,'--output',base,'--target-rank',7,'--target-fraction',.1,'--no-base-counts','--uncompressed-maps'])
previous=None
for suffix in ['price_f2','price_b8','price_b32','guard_p5']:
 params=json.loads((old/('rapid_'+suffix)/'refinement.json').read_text())['parameters'];dest=baseout/('rapid_'+suffix)
 cmd=[P,S/'joint_price_owners.py','--case',base,'--cache',baseout/'profile_cache','--output',dest]
 for key in ['forward','backward','storage','penalty','budget','passes','hot_threshold']:
  if key in params:cmd+=['--'+key.replace('_','-'),params[key]]
 if previous:cmd+=['--resume-from',previous]
 if suffix=='guard_p5':cmd+=['--guard-case',previous]
 run(suffix,cmd);previous=dest
run('optimized',[P,X/'run_optimized.py','--ranks',g,'--baseline',baseout,'--gmp-meta',out/'gmp_traffic/het_gmp_s0.json','--out',out/'optimized'])
save(out/'complete.json',dict(success=True,finished=time.time()))
