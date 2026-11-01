"""Fresh 16-rank preprocessing and separately timed logical communication replay."""
import hashlib,json,os,sys,time
from pathlib import Path
from common import measured,save
ROOT=Path(__file__).resolve().parents[3];REP=ROOT/'reproduction';EXP=REP/'het_gmp_rapid'
OUT=EXP/'results/rank16_fast';OLD=EXP/'results/timing8_rerun';S=REP/'scripts';WORK=REP/'working';PY=sys.executable;DATA=OUT/'data'
def run(n,cmd):return measured(n,cmd,OUT,threads=16)
def replace(cmd,values):
 cmd=list(cmd)
 for k,v in values.items():cmd[cmd.index(k)+1]=str(v)
 return cmd
def main():
 OUT.mkdir(exist_ok=False)
 os.environ['LD_LIBRARY_PATH']=str(REP/'toolchain/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
 save(OUT/'benchmark_config.json',dict(rows=4194304,ranks=16,local_batch=8192,batches=32,epochs=5,omp_threads=16,blas_threads=1,batch_workers=16,
 rapid='Current verified eight-rank workflow transferred with fixed parameters to 16 ranks; target7 fraction.1 initial owner, f2/b8/b32/guard_p5 chain; no new tuning',
 gmp='Native upstream 10 partition rounds; .01 mirror candidate fraction; bound0 CPU protocol replay',
 topology='Uniform off-diagonal cost .1; logical ranks, not a measured physical fabric',
 timing='Sequential fresh generation; normal OS cache. Common source slicing excluded equally. Replay and post-run output verification excluded.',
 traffic='Remote embedding/gradient vector counts only; excludes metadata, local accesses and dense allreduce. No GPU/NCCL measurement. All samples and tail batches retained in totals.'))
 sources=[Path(__file__),WORK/'cps/generate_local_dataset',WORK/'cps/generate_local_dataset.cpp',WORK/'cps/swap_optimizer.h',S/'refine_trace_owners.py',S/'joint_price_owners.py',S/'joint_price_owners.so',S/'joint_price_guarded.so',S/'refine_balanced_owners.py',EXP/'scripts/replay_het.py',EXP/'scripts/partition_het.py',EXP/'scripts/hetuCTR_partition.cpython-310-x86_64-linux-gnu.so']
 sources+=list((EXP/'src/het-gmp/hetuCTR/csrc/core').glob('*.cu'))
 save(OUT/'source_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 run('input_and_gmp_adapter',[PY,EXP/'scripts/prepare.py','--input',ROOT/'data/processed/criteo/day6.bin','--out',DATA,'--batches',32,'--ranks',16])
 assert json.loads((DATA/'input.json').read_text())['input_sha256']=='44ba74ba68f60c13d36d3724cbb7dd95b30d997fc9d489cbd67347dc3fe70317'
 run('gmp_partition',[PY,EXP/'scripts/partition_het.py','--data',DATA,'--out',OUT/'gmp','--ranks',16])
 run('gmp_materialize',[PY,EXP/'scripts/materialize_het.py','--data',DATA,'--partition',OUT/'gmp','--ranks',16])
 model=OUT/'rapid_schedule';model.mkdir()
 cfg=json.loads((OLD/'rapid_schedule/config.json').read_text());cfg.update(input=str(DATA/'input.bin'),output=str(model),base_case=None,ranks=16,batches=32,rows=4194304)
 save(model/'config.json',cfg)
 cmd=replace(json.loads((OLD/'rapid_hot.command.json').read_text()),{'--input':DATA/'input.bin','--output':model/'hot'});run('rapid_hot',cmd)
 cmd=replace(json.loads((OLD/'rapid_placement.command.json').read_text()),{'--input_bin':DATA/'input.bin','--hot_ids_dir':model/'hot','--output_dir':model/'placement','--num_devices':16,'--n_batches':32});run('rapid_placement',cmd)
 cmd=replace(json.loads((OLD/'rapid_schedule.command.json').read_text()),{'--input_file':DATA/'input.bin','--table_files_dir':model/'placement','--output_dir':model/'scheduled','--num_devices':16,'--num_chunks':32,'--num_threads':16});cmd[0]=str(WORK/'cps/generate_local_dataset');run('rapid_schedule',cmd)
 base=OUT/'rapid_owner_base';run('rapid_owner_base',[PY,S/'refine_trace_owners.py','--case',model,'--output',base,'--target-rank',7,'--target-fraction',.1,'--no-base-counts','--uncompressed-maps'])
 previous=None
 for suffix in ['price_f2','price_b8','price_b32','guard_p5']:
  params=json.loads((OLD/('rapid_'+suffix)/'refinement.json').read_text())['parameters'];dest=OUT/('rapid_'+suffix)
  cmd=[PY,S/'joint_price_owners.py','--case',base,'--cache',OUT/'rapid_profile_cache','--output',dest]
  for key in ['forward','backward','storage','penalty','budget','passes','hot_threshold']:
   if key in params:cmd+=['--'+key.replace('_','-'),params[key]]
  if previous:cmd+=['--resume-from',previous]
  if suffix=='guard_p5':cmd+=['--guard-case',previous]
  run('rapid_'+suffix,cmd);previous=dest
 run('rapid_final_materialize',[PY,S/'refine_balanced_owners.py','--case',base,'--cache',OUT/'rapid_profile_cache','--output',previous,'--materialize-existing','--uncompressed-maps'])
 save(OUT/'preprocess_complete.json',dict(success=True,completed=time.time()))
 print('PREPROCESSING COMPLETE; replay is separately timed',flush=True)
 run('gmp_replay',[PY,EXP/'scripts/replay_het.py','--data',DATA,'--partition',OUT/'gmp','--out',OUT/'traffic','--ranks',16,'--epochs',5,'--bounds',0,'--skip-1d'])
 run('verify_outputs',[PY,EXP/'scripts/verify_rank16.py'])
 run('report',[PY,EXP/'scripts/report_rank16.py'])
 print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
