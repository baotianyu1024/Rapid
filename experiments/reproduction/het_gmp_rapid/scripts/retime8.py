"""Fresh sequential eight-rank generation, including the selected RAPID parent chain."""
import hashlib,json,os,sys,time
from pathlib import Path
from common import measured,save,LENGTHS
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'reproduction/het_gmp_rapid';OUT=EXP/'results/timing8_rerun'
OLD=ROOT/'reproduction/results';SCRIPTS=ROOT/'reproduction/scripts';WORK=ROOT/'reproduction/working'
DATA=OUT/'data';PY=sys.executable

def run(name,cmd):return measured(name,cmd,OUT,threads=16)
def replace(cmd,values):
 cmd=list(cmd)
 for flag,value in values.items():cmd[cmd.index(flag)+1]=str(value)
 return cmd

def main():
 assert not (OUT/'input_and_gmp_adapter.command.json').exists()
 assert json.loads((OUT/'cleanup_manifest.json').read_text())['completed']
 os.environ['LD_LIBRARY_PATH']=str(ROOT/'reproduction/toolchain/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
 save(OUT/'benchmark_config.json',dict(rows=4194304,ranks=8,local_batch=8192,batches=64,order=['GMP','RAPID'],omp_threads=16,openblas_threads=1,
 rapid='accelerated exact scheduler + original owner tie target7/fraction.1 + selected f2,b8,b32,guard_p5 chain; fresh profile cache',
 gmp_rounds=10,gmp_store_rate=.01,cache_policy='normal filesystem cache; no system drop_caches',
 maps='final RAPID maps stored uncompressed; intermediate original-owner maps retain original compression',
 excludes='installation, compilation, historical hyperparameter search, optional post-run validation',
 includes='fresh ID encoding/profile generation, required warm-start passes, full label+dense+sparse output; internal validation included and identified'))
 sources=[Path(__file__),WORK/'cps/generate_local_dataset',WORK/'cps/generate_local_dataset.cpp',WORK/'cps/swap_optimizer.h',WORK/'cps/flat_counts.h',SCRIPTS/'joint_price_owners.so',SCRIPTS/'joint_price_guarded.so',SCRIPTS/'joint_price_owners.py',SCRIPTS/'refine_balanced_owners.py',SCRIPTS/'refine_trace_owners.py',EXP/'scripts/partition_het.py',EXP/'scripts/hetuCTR_partition.cpython-310-x86_64-linux-gnu.so',ROOT/'reproduction/baseline/update_Rapid/cps/arrange_ids_to_devices']
 save(OUT/'source_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 run('input_and_gmp_adapter',[PY,EXP/'scripts/prepare.py','--input',ROOT/'data/processed/criteo/day6.bin','--out',DATA,'--batches',64,'--ranks',8])
 assert json.loads((DATA/'input.json').read_text())['input_sha256']=='44ba74ba68f60c13d36d3724cbb7dd95b30d997fc9d489cbd67347dc3fe70317'
 run('gmp_partition',[PY,EXP/'scripts/partition_het.py','--data',DATA,'--out',OUT/'gmp','--ranks',8])
 run('gmp_materialize',[PY,EXP/'scripts/materialize_het.py','--data',DATA,'--partition',OUT/'gmp','--ranks',8])
 model=OUT/'rapid_schedule';model.mkdir()
 cfg=json.loads((OLD/'criteo_day6_8_updated_64b_improved/config.json').read_text())
 cfg.update(input=str(DATA/'input.bin'),output=str(model),base_case=None,repo=str(WORK))
 save(model/'config.json',cfg)
 cmd=json.loads((OLD/'criteo_day6_8_updated_64b/hot.command.json').read_text());cmd[0]=PY
 cmd=replace(cmd,{'--input':DATA/'input.bin','--output':model/'hot'})
 run('rapid_hot',cmd)
 cmd=json.loads((OLD/'criteo_day6_8_updated_64b/placement.command.json').read_text())
 cmd=replace(cmd,{'--input_bin':DATA/'input.bin','--hot_ids_dir':model/'hot','--output_dir':model/'placement'})
 run('rapid_placement',cmd)
 cmd=json.loads((OLD/'criteo_day6_8_updated_64b_improved/schedule.command.json').read_text())
 cmd[0]=str(WORK/'cps/generate_local_dataset')
 cmd=replace(cmd,{'--input_file':DATA/'input.bin','--table_files_dir':model/'placement','--output_dir':model/'scheduled'})
 run('rapid_schedule',cmd)
 base=OUT/'rapid_owner_base'
 run('rapid_owner_base',[PY,SCRIPTS/'refine_trace_owners.py','--case',model,'--output',base,'--target-rank',7,'--target-fraction',.1,'--no-base-counts'])
 # The current best policy has three required warm-start stages. No historical arrays are reused.
 previous=None
 for suffix in ['price_f2','price_b8','price_b32','guard_p5']:
  params=json.loads((OLD/('criteo_day6_8_'+suffix)/'refinement.json').read_text())['parameters']
  dest=OUT/('rapid_'+suffix)
  cmd=[PY,SCRIPTS/'joint_price_owners.py','--case',base,'--cache',OUT/'rapid_profile_cache','--output',dest]
  for key in ['forward','backward','storage','penalty','budget','passes','hot_threshold']:
   if key in params:cmd+=['--'+key.replace('_','-'),params[key]]
  if previous:cmd+=['--resume-from',previous]
  if suffix=='guard_p5':cmd+=['--guard-case',previous]
  run('rapid_'+suffix,cmd);previous=dest
 run('rapid_final_materialize',[PY,SCRIPTS/'refine_balanced_owners.py','--case',base,'--cache',OUT/'rapid_profile_cache','--output',previous,'--materialize-existing','--uncompressed-maps'])
 save(OUT/'preprocess_complete.json',dict(success=True,completed=time.time()))
 print('ALL PREPROCESSING COMPLETE',flush=True)
if __name__=='__main__':main()
