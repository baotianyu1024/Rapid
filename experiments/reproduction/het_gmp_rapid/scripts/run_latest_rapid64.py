"""Default RAPID for 64-rank experiments: snapshot and compile current sources."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
from common import measured,save,LENGTHS
EXP=Path(__file__).resolve().parents[1];REP=EXP.parent
WORK_FILES=['cps/arrange_ids_to_devices.cpp','cps/generate_local_dataset.cpp','cps/swap_optimizer.h','cps/flat_counts.h','cps/hot_id_extraction.py','acp/compute_acp.py']
SCRIPT_FILES=['refine_trace_owners.py','joint_price_owners.py','refine_balanced_owners.py','measure.py','joint_price_owners.cpp','joint_price_guarded.cpp']
CHAIN=[('price_f2',2,1,30,.2,4,0),('price_b8',4,8,30,.5,3,32),('price_b32',8,32,60,.5,3,16),('guard_p5',8,32,60,5,3,16)]
def validate_manifest(path):
 manifest=json.loads(Path(path).read_text())
 expected={str(REP/'working'/p) for p in WORK_FILES}|{str(REP/'scripts'/p) for p in SCRIPT_FILES}
 assert set(manifest)==expected,'RAPID manifest must include the complete current-source workflow'
 for src,v in manifest.items():
  digest=hashlib.sha256(Path(src).read_bytes()).hexdigest()
  assert digest==v['sha256'],f'Current RAPID source changed: {src}; use a fresh --out directory'
  assert hashlib.sha256(Path(v['snapshot']).read_bytes()).hexdigest()==digest,'Snapshot mismatch'
 return manifest

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=EXP/'results/rank64_balance');a=p.parse_args();case=a.case.resolve()
 out=case/'latest_rapid';src=case/'latest_source';data=case/'data'
 out.mkdir(exist_ok=False);src.mkdir(exist_ok=False)
 work=src/'working';scripts=src/'scripts';manifest={}
 for root,files,destroot in [(REP/'working',WORK_FILES,work),(REP/'scripts',SCRIPT_FILES,scripts)]:
  for rel in files:
   source=root/rel;dest=destroot/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
   manifest[str(source)]={'snapshot':str(dest),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
 save(out/'source_manifest.json',manifest);validate_manifest(out/'source_manifest.json')
 def run(name,cmd):return measured(name,cmd,out,threads=16)
 cxx=Path(sys.prefix)/'bin/x86_64-conda-linux-gnu-g++'
 for name in ['arrange_ids_to_devices','generate_local_dataset']:
  run('build_'+name,[cxx,'-O3','-std=c++17','-fopenmp','-pthread',work/'cps'/f'{name}.cpp','-o',work/'cps'/name])
 for name in ['joint_price_owners','joint_price_guarded']:
  run('build_'+name,[cxx,'-O3','-march=native','-std=c++17','-shared','-fPIC',scripts/f'{name}.cpp','-o',scripts/f'{name}.so'])
 model=out/'rapid_schedule';model.mkdir()
 save(model/'config.json',dict(input=str(data/'input.bin'),dataset='criteo',repo=str(work),output=str(model),base_case=None,ranks=64,batches=8,rows=4194304,batch_size=8192,dense=13,table_lengths=LENGTHS,topk=40000,swap='ratio',strict_capacity=False,gpu_execution=False,scope='same day6 prefix, complete global batches'))
 run('hot',[sys.executable,work/'cps/hot_id_extraction.py','--input',data/'input.bin','--output',model/'hot','--topk',40000,'--max_rows',4194304,'--table_lengths',*LENGTHS,'--dense_count',13])
 run('placement',[work/'cps/arrange_ids_to_devices','--input_bin',data/'input.bin','--hot_ids_dir',model/'hot','--output_dir',model/'placement','--batch_size',8192,'--num_devices',64,'--n_batches',8,'--table_lengths',*LENGTHS,'--dense_count',13])
 run('schedule',[work/'cps/generate_local_dataset','--input_file',data/'input.bin','--table_files_dir',model/'placement','--output_dir',model/'scheduled','--num_devices',64,'--train_batch_size',8192,'--num_chunks',8,'--steps_per_chunk',1,'--num_threads',8,'--gamma',.2,'--dense_count',13,'--table_lengths',*LENGTHS,'--swap_algorithm','ratio','--swap_screen_queries_per_gpu',20000000,'--swap_evaluations_per_gpu',500000])
 run('sample_audit',[EXP/'scripts/audit_samples64',data/'input.bin',model,model/'sample_audit.json'])
 base=out/'rapid_owner_base';run('owner_base',[sys.executable,scripts/'refine_trace_owners.py','--case',model,'--output',base,'--target-rank',7,'--target-fraction',.1,'--no-base-counts','--uncompressed-maps'])
 previous=None
 for suffix,f,b,s,penalty,passes,hot in CHAIN:
  dest=out/('rapid_'+suffix);cmd=[sys.executable,scripts/'joint_price_owners.py','--case',base,'--cache',out/'profile_cache','--output',dest,'--forward',f,'--backward',b,'--storage',s,'--penalty',penalty,'--budget',.05,'--passes',passes,'--hot-threshold',hot]
  if previous:cmd+=['--resume-from',previous]
  if suffix=='guard_p5':cmd+=['--guard-case',previous]
  run(suffix,cmd);previous=dest
 run('materialize',[sys.executable,scripts/'refine_balanced_owners.py','--case',base,'--cache',out/'profile_cache','--output',previous,'--materialize-existing','--uncompressed-maps'])
 run('acp',[sys.executable,work/'acp/compute_acp.py','--input',data/'input.bin','--num_gpus',64,'--batch_size',8192,'--num_batches',8,'--cps_dataset_dirs',previous/'scheduled','--output',previous/'acp.json','--dense_count',13,'--table_lengths',*LENGTHS])
 acp=json.loads((previous/'acp.json').read_text())['64'];assert all(t['strategy']=='Unique' for t in acp['acp_cps_per_table'])
 validate_manifest(out/'source_manifest.json')
 save(out/'complete.json',dict(success=True,source='current reproduction/working + reproduction/scripts snapshot',recipe='PARALLEL_RAPID_RUNBOOK fixed owner chain; no retuning',final_case=str(previous),all_tables_unique=True))
if __name__=='__main__':main()
