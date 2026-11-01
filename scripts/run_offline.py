"""Fresh RAPID baseline using the final revision experiment's owner/sample recipe."""
import argparse,hashlib,json,os,resource,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'experiments/reproduction';X=R/'rapid_peak_optimization';S=R/'scripts';H=R/'het_gmp_rapid/scripts'
L=[3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
 p.add_argument('--ranks',type=int,choices=[8,64],default=8);p.add_argument('--batch-size',type=int,default=8192);p.add_argument('--batches',type=int,default=16)
 p.add_argument('--dataset',choices=['criteo','kaggle','taobao','custom'],default='criteo');p.add_argument('--dense',type=int);p.add_argument('--table-lengths',type=int,nargs='+');p.add_argument('--topk',type=int,default=40000)
 p.add_argument('--plan-only',action='store_true');a=p.parse_args()
 if not 1<=a.batches<=64:p.error('Offline recipe supports 1..64 global batches; wide-window incremental uses a separate entrypoint.')
 if a.batch_size<1 or a.topk<1:p.error('batch-size and topk must be positive')
 if a.dataset=='custom' and (a.dense is None or not a.table_lengths):p.error('custom requires --dense and --table-lengths')
 lengths=a.table_lengths or ([1]*4 if a.dataset=='taobao' else L);dense=a.dense if a.dense is not None else (1 if a.dataset=='taobao' else 13)
 if dense<0 or any(x<1 for x in lengths):p.error('Invalid record layout')
 out=a.out.resolve();src=a.input.resolve();g=a.ranks;b=a.batch_size;n=a.batches*g*b
 baseout=out/'baseline';model=baseout/'rapid_schedule';base=baseout/'rapid_owner_base';data=out/'data';commands=[]
 def add(name,args):commands.append((name,list(map(str,args))))
 add('prepare',[sys.executable,H/'prepare.py','--input',src,'--out',data,'--batches',a.batches,'--ranks',g,'--batch-size',b,'--dataset',a.dataset,'--dense',dense,'--table-lengths',*lengths])
 common=['--dense_count',dense,'--table_lengths',*lengths]
 add('hot',[sys.executable,ROOT/'cps/hot_id_extraction.py','--input',data/'input.bin','--output',model/'hot','--topk',a.topk,'--max_rows',n,*common])
 add('placement',[ROOT/'cps/arrange_ids_to_devices','--input_bin',data/'input.bin','--hot_ids_dir',model/'hot','--output_dir',model/'placement','--batch_size',b,'--num_devices',g,'--n_batches',a.batches,*common])
 add('schedule',[ROOT/'cps/generate_local_dataset','--input_file',data/'input.bin','--table_files_dir',model/'placement','--output_dir',model/'scheduled','--num_devices',g,'--train_batch_size',b,'--num_chunks',a.batches,'--steps_per_chunk',1,'--num_threads',8,'--gamma',.2,'--swap_algorithm','ratio','--swap_screen_queries_per_gpu',20000000,'--swap_evaluations_per_gpu',500000,*common])
 add('owner_base',[sys.executable,S/'refine_trace_owners.py','--case',model,'--output',base,'--target-rank',7,'--target-fraction',.1,'--no-base-counts','--uncompressed-maps'])
 previous=None
 for suffix,f,bw,storage,pen,budget,passes,hot in [('price_f2',2,1,30,.2,.05,4,0),('price_b8',4,8,30,.5,.05,3,32),('price_b32',8,32,60,.5,.05,3,16),('guard_p5',8,32,60,5,.05,3,16)]:
  dest=baseout/('rapid_'+suffix);cmd=[sys.executable,S/'joint_price_owners.py','--case',base,'--cache',baseout/'profile_cache','--output',dest,'--forward',f,'--backward',bw,'--storage',storage,'--penalty',pen,'--budget',budget,'--passes',passes,'--hot-threshold',hot,'--allow-reduction-tables']
  if previous:cmd+=['--resume-from',previous]
  if suffix=='guard_p5':cmd+=['--guard-case',previous]
  add(suffix,cmd);previous=dest
 add('rapid',[sys.executable,X/'run_optimized_kaggle.py','--ranks',g,'--baseline',baseout,'--out',out/'optimized'])
 add('summary',[sys.executable,ROOT/'scripts/summarize.py','--case',out/'optimized/peak_samples','--out',out/'communication.json'])
 if a.plan_only:print(json.dumps(dict(recipe='RAPID final revision offline',commands=commands),indent=2));return
 if not src.is_file() or src.stat().st_size<4*(1+dense+sum(lengths))*n:p.error('Input missing or too short for requested complete global batches')
 if out.exists():p.error('Use a fresh output directory; existing outputs are never overwritten')
 soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard) if hard!=-1 else 65536,hard))
 model.mkdir(parents=True)
 cfg=dict(dataset=a.dataset,input=str(data/'input.bin'),output=str(model),repo=str(ROOT),ranks=g,batches=a.batches,rows=n,batch_size=b,dense=dense,table_lengths=lengths,topk=a.topk,hot_rows=n,base_case=None,gpu_execution=False)
 (model/'config.json').write_text(json.dumps(cfg,indent=2))
 sources=[f for d in [S,X,H,ROOT/'cps'] for f in d.iterdir() if f.is_file() and f.suffix in ['.py','.cpp','.h','.so']]
 (baseout/'source_manifest.json').write_text(json.dumps({str(f):dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sources},indent=2))
 (out/'experiment.json').write_text(json.dumps(dict(**cfg,dim=128,bytes_per_component=4,epochs=5,recipe='RAPID final revision offline'),indent=2))
 for name,cmd in commands:
  print('START',name,flush=True);(out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2))
  with (out/(name+'.log')).open('w') as log:
   proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8'))
  if proc.returncode:raise RuntimeError(f'{name} failed: {out / (name+".log")}')
 (out/'complete.json').write_text(json.dumps(dict(success=True,final_case=str(out/'optimized/peak_samples'),gpu_execution=False),indent=2))
 print('COMPLETE',out)
if __name__=='__main__':main()
