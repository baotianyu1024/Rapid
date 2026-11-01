"""Fresh RAPID pipeline with ordered concurrent batches and exact LUT remapping."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'het_gmp_rapid';OLD=EXP/'results/timing8_rerun';OUT=ROOT/'results/speed_parallel8/run'
sys.path.insert(0,str(EXP/'scripts'))
from common import measured,save
PY=sys.executable;S=ROOT/'scripts';DATA=OLD/'data';WORK=ROOT/'working'
def run(n,cmd):return measured(n,cmd,OUT,threads=16)
def replace(cmd,values):
 cmd=list(cmd)
 for key,value in values.items():cmd[cmd.index(key)+1]=str(value)
 return cmd
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=False);os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
 save(OUT/'benchmark_config.json',dict(rows=4194304,ranks=8,batches=64,local_batch=8192,omp_budget=16,batch_workers=16,worker_omp_threads=1,input=str(DATA/'input.bin'),scope='Fresh RAPID hot IDs, placement, all64batch scheduling, owner base, fresh profile, fixed four-stage owner chain, final materialization; common input slice reused exactly as prior timing boundary; no other arrays reused',reference=str(OLD),maps='Uncompressed mapping archives with identical arrays',validation='Generation includes internal checks; final historical byte/map comparison excluded'))
 model=OUT/'rapid_schedule';model.mkdir();cfg=json.loads((OLD/'rapid_schedule/config.json').read_text());cfg.update(output=str(model),base_case=None);save(model/'config.json',cfg)
 cmd=replace(json.loads((OLD/'rapid_hot.command.json').read_text()),{'--output':model/'hot'});run('rapid_hot',cmd)
 cmd=replace(json.loads((OLD/'rapid_placement.command.json').read_text()),{'--hot_ids_dir':model/'hot','--output_dir':model/'placement'});run('rapid_placement',cmd)
 cmd=replace(json.loads((OLD/'rapid_schedule.command.json').read_text()),{'--table_files_dir':model/'placement','--output_dir':model/'scheduled','--num_threads':16});cmd[0]=str(ROOT/'speed_work/parallel/scheduler')
 save(OUT/'source_sha256.json',{str(p):sha(p) for p in [Path(cmd[0]),WORK/'cps/generate_local_dataset.cpp',WORK/'cps/swap_optimizer.h',WORK/'cps/flat_counts.h',S/'refine_trace_owners.py',S/'joint_price_owners.py',S/'joint_price_owners.so',S/'joint_price_guarded.so',S/'refine_balanced_owners.py',Path(__file__)]})
 run('rapid_schedule',cmd)
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
 print('GENERATION COMPLETE; verifying historical output',flush=True)
 import numpy as np
 checks={}
 for stage in ['rapid_schedule','rapid_owner_base','rapid_guard_p5']:
  checks[stage]=[sha(OUT/stage/f'scheduled/dataset_{r}.bin')==sha(OLD/stage/f'scheduled/dataset_{r}.bin') for r in range(8)]
  assert all(checks[stage]),stage
  print(stage,'all8shards exact',flush=True)
 for stage in ['rapid_owner_base','rapid_guard_p5']:
  for t in range(26):
   with np.load(OUT/stage/f'table_{t}_remap.npz') as a,np.load(OLD/stage/f'table_{t}_remap.npz') as b:
    assert set(a.files)==set(b.files)
    for k in a.files:assert np.array_equal(a[k],b[k]),(stage,t,k)
 save(OUT/'verification.json',dict(success=True,all_output_bytes_equal=True,all_mapping_arrays_equal=True,shards=checks))
 print('ALL VERIFIED',flush=True)
if __name__=='__main__':main()
