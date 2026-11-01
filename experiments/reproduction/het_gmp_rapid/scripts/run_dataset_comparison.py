"""Fresh 16-rank preprocessing and separately timed logical communication replay."""
import hashlib,json,os,sys,time,argparse
from pathlib import Path
from common import measured,save
ROOT=Path(__file__).resolve().parents[3];REP=ROOT/'reproduction';EXP=REP/'het_gmp_rapid'
ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['taobao','kaggle'],required=True);a=ap.parse_args()
DS=a.dataset;LENGTHS=[1]*4 if DS=='taobao' else [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1];DENSE=1 if DS=='taobao' else 13
INPUT=ROOT/('data/processed/taobao/taobao_4tables.bin' if DS=='taobao' else 'data/processed/kaggle/multihot.bin')
OUT=EXP/('results/'+DS+'8_fast');OLD=EXP/'results/timing8_rerun';S=REP/'scripts';WORK=REP/'working';PY=sys.executable;DATA=OUT/'data'
def run(n,cmd):return measured(n,cmd,OUT,threads=16)
def replace(cmd,values):
 cmd=list(cmd)
 if '--table_lengths' in cmd:
  i=cmd.index('--table_lengths')+1;j=i
  while j<len(cmd) and not cmd[j].startswith('--'):j+=1
  cmd[i:j]=[str(x) for x in LENGTHS]
 if '--dense_count' in cmd:cmd[cmd.index('--dense_count')+1]=str(DENSE)
 for k,v in values.items():cmd[cmd.index(k)+1]=str(v)
 return cmd
def main():
 OUT.mkdir(exist_ok=False)
 os.environ['LD_LIBRARY_PATH']=str(REP/'toolchain/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
 save(OUT/'benchmark_config.json',dict(rows=4194304,ranks=8,local_batch=8192,batches=64,epochs=5,omp_threads=16,blas_threads=1,batch_workers=16,
 dataset=DS,rapid='Verified eight-rank workflow; fixed Criteo parameters transferred to '+DS+'; no new tuning',
 gmp='Native upstream 10 partition rounds; .01 mirror candidate fraction; bound0 CPU protocol replay',
 topology='Uniform off-diagonal cost .1; logical ranks, not a measured physical fabric',
 timing='Sequential fresh generation; normal OS cache. Common source slicing excluded equally. Replay and post-run output verification excluded.',
 traffic='Remote embedding/gradient vector counts only; excludes metadata, local accesses and dense allreduce. No GPU/NCCL measurement. All samples and tail batches retained in totals.'))
 sources=[EXP/'scripts/prepare.py',EXP/'scripts/materialize_het.py',EXP/'scripts/verify_dataset.py',EXP/'scripts/report_dataset.py',Path(__file__),WORK/'cps/generate_local_dataset',WORK/'cps/generate_local_dataset.cpp',WORK/'cps/swap_optimizer.h',S/'refine_trace_owners.py',S/'joint_price_owners.py',S/'joint_price_owners.so',S/'joint_price_guarded.so',S/'refine_balanced_owners.py',EXP/'scripts/replay_het.py',EXP/'scripts/partition_het.py',EXP/'scripts/hetuCTR_partition.cpython-310-x86_64-linux-gnu.so']
 sources+=list((EXP/'src/het-gmp/hetuCTR/csrc/core').glob('*.cu'))
 save(OUT/'source_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 run('input_and_gmp_adapter',[PY,EXP/'scripts/prepare.py','--input',INPUT,'--out',DATA,'--batches',64,'--ranks',8,'--dataset',DS,'--dense',DENSE,'--table-lengths',*LENGTHS])
 run('gmp_partition',[PY,EXP/'scripts/partition_het.py','--data',DATA,'--out',OUT/'gmp','--ranks',8])
 run('gmp_materialize',[PY,EXP/'scripts/materialize_het.py','--data',DATA,'--partition',OUT/'gmp','--ranks',8])
 model=OUT/'rapid_schedule';model.mkdir()
 cfg=json.loads((OLD/'rapid_schedule/config.json').read_text());cfg.update(input=str(DATA/'input.bin'),output=str(model),base_case=None,ranks=8,batches=64,rows=4194304)
 cfg.update(dataset=DS,dense=DENSE,table_lengths=LENGTHS,input_total_rows=INPUT.stat().st_size//(4*(1+DENSE+sum(LENGTHS))),input_metadata={'scope':'existing shuffled '+DS},hot_rows=4194304)
 save(model/'config.json',cfg)
 cmd=replace(json.loads((OLD/'rapid_hot.command.json').read_text()),{'--input':DATA/'input.bin','--output':model/'hot'});run('rapid_hot',cmd)
 cmd=replace(json.loads((OLD/'rapid_placement.command.json').read_text()),{'--input_bin':DATA/'input.bin','--hot_ids_dir':model/'hot','--output_dir':model/'placement','--num_devices':8,'--n_batches':64});run('rapid_placement',cmd)
 cmd=replace(json.loads((OLD/'rapid_schedule.command.json').read_text()),{'--input_file':DATA/'input.bin','--table_files_dir':model/'placement','--output_dir':model/'scheduled','--num_devices':8,'--num_chunks':64,'--num_threads':16});cmd[0]=str(WORK/'cps/generate_local_dataset');run('rapid_schedule',cmd)
 base=OUT/'rapid_owner_base';run('rapid_owner_base',[PY,S/'refine_trace_owners.py','--case',model,'--output',base,'--target-rank',7,'--target-fraction',.1,'--no-base-counts','--uncompressed-maps'])
 previous=None
 for suffix in ['price_f2','price_b8','price_b32','guard_p5']:
  params=json.loads((OLD/('rapid_'+suffix)/'refinement.json').read_text())['parameters'];dest=OUT/('rapid_'+suffix)
  cmd=[PY,S/'joint_price_owners.py','--case',base,'--cache',OUT/'rapid_profile_cache','--output',dest,'--allow-reduction-tables']
  for key in ['forward','backward','storage','penalty','budget','passes','hot_threshold']:
   if key in params:cmd+=['--'+key.replace('_','-'),params[key]]
  if previous:cmd+=['--resume-from',previous]
  if suffix=='guard_p5':cmd+=['--guard-case',previous]
  run('rapid_'+suffix,cmd);previous=dest
 run('rapid_final_materialize',[PY,S/'refine_balanced_owners.py','--case',base,'--cache',OUT/'rapid_profile_cache','--output',previous,'--materialize-existing','--uncompressed-maps'])
 save(OUT/'preprocess_complete.json',dict(success=True,completed=time.time()))
 print('PREPROCESSING COMPLETE; replay is separately timed',flush=True)
 run('gmp_replay',[PY,EXP/'scripts/replay_het.py','--data',DATA,'--partition',OUT/'gmp','--out',OUT/'traffic','--ranks',8,'--epochs',5,'--bounds',0,'--skip-1d'])
 run('verify_outputs',[PY,EXP/'scripts/verify_dataset.py','--case',OUT])
 run('report',[PY,EXP/'scripts/report_dataset.py','--case',OUT])
 print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
