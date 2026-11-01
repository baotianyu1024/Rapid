"""Fresh current working sources + latest documented full owner refinement chain."""
import hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT.parent; EXP=REP/'het_gmp_rapid'; OLD=EXP/'results/timing8_rerun'
OUT=ROOT/'results/default8/latest_rapid';SRC=ROOT/'latest_source';DATA=EXP/'results/main64/data'
sys.path.insert(0,str(EXP/'scripts'))
from common import measured,save,LENGTHS

def run(name,cmd):return measured(name,cmd,OUT,threads=16)
def replace(cmd,values):
    cmd=list(cmd)
    for k,v in values.items():cmd[cmd.index(k)+1]=str(v)
    return cmd
def main():
    OUT.mkdir(exist_ok=False);SRC.mkdir(exist_ok=False)
    work=SRC/'working';scripts=SRC/'scripts'
    files=['cps/arrange_ids_to_devices.cpp','cps/generate_local_dataset.cpp','cps/swap_optimizer.h','cps/flat_counts.h','cps/hot_id_extraction.py','acp/compute_acp.py']
    manifest={}
    for rel in files:
        source=REP/'working'/rel;dest=work/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        manifest[str(source)]={'snapshot':str(dest),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    for rel in ['refine_trace_owners.py','joint_price_owners.py','refine_balanced_owners.py','measure.py','joint_price_owners.cpp','joint_price_guarded.cpp']:
        source=REP/'scripts'/rel;dest=scripts/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        manifest[str(source)]={'snapshot':str(dest),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    save(OUT/'source_manifest.json',manifest)
    cxx=Path(sys.prefix)/'bin/x86_64-conda-linux-gnu-g++'
    for name in ['arrange_ids_to_devices','generate_local_dataset']:
        run('build_'+name,[cxx,'-O3','-std=c++17','-fopenmp','-pthread',work/'cps'/f'{name}.cpp','-o',work/'cps'/name])
    for name in ['joint_price_owners','joint_price_guarded']:
        run('build_'+name,[cxx,'-O3','-march=native','-std=c++17','-shared','-fPIC',scripts/f'{name}.cpp','-o',scripts/f'{name}.so'])
    model=OUT/'rapid_schedule';model.mkdir()
    cfg=json.loads((OLD/'rapid_schedule/config.json').read_text());cfg.update(input=str(DATA/'input.bin'),repo=str(work),output=str(model),base_case=None)
    save(model/'config.json',cfg)
    cmd=replace(json.loads((OLD/'rapid_hot.command.json').read_text()),{'--input':DATA/'input.bin','--output':model/'hot'});cmd[0]=sys.executable;cmd[1]=str(work/'cps/hot_id_extraction.py');run('hot',cmd)
    cmd=replace(json.loads((OLD/'rapid_placement.command.json').read_text()),{'--input_bin':DATA/'input.bin','--hot_ids_dir':model/'hot','--output_dir':model/'placement'});cmd[0]=str(work/'cps/arrange_ids_to_devices');run('placement',cmd)
    cmd=replace(json.loads((OLD/'rapid_schedule.command.json').read_text()),{'--input_file':DATA/'input.bin','--table_files_dir':model/'placement','--output_dir':model/'scheduled','--num_threads':16});cmd[0]=str(work/'cps/generate_local_dataset');run('schedule',cmd)
    run('sample_audit',[EXP/'scripts/audit_samples',DATA/'input.bin',model,model/'sample_audit.json'])
    base=OUT/'rapid_owner_base'
    run('owner_base',[sys.executable,scripts/'refine_trace_owners.py','--case',model,'--output',base,'--target-rank',7,'--target-fraction',.1,'--no-base-counts','--uncompressed-maps'])
    previous=None
    for suffix in ['price_f2','price_b8','price_b32','guard_p5']:
        params=json.loads((OLD/('rapid_'+suffix)/'refinement.json').read_text())['parameters'];dest=OUT/('rapid_'+suffix)
        cmd=[sys.executable,scripts/'joint_price_owners.py','--case',base,'--cache',OUT/'profile_cache','--output',dest]
        for key in ['forward','backward','storage','penalty','budget','passes','hot_threshold']:
            if key in params:cmd+=['--'+key.replace('_','-'),params[key]]
        if previous:cmd+=['--resume-from',previous]
        if suffix=='guard_p5':cmd+=['--guard-case',previous]
        run(suffix,cmd);previous=dest
    run('materialize',[sys.executable,scripts/'refine_balanced_owners.py','--case',base,'--cache',OUT/'profile_cache','--output',previous,'--materialize-existing','--uncompressed-maps'])
    run('acp',[sys.executable,work/'acp/compute_acp.py','--input',DATA/'input.bin','--num_gpus',8,'--batch_size',8192,'--num_batches',50,'--cps_dataset_dirs',previous/'scheduled','--output',previous/'acp.json','--dense_count',13,'--table_lengths',*LENGTHS])
    # All tables are verified Unique by materialize(); the old replay counter's
    # original-vs-remote Reduction threshold therefore cannot alter this result.
    acp=json.loads((previous/'acp.json').read_text())['8']
    assert all(t['strategy']=='Unique' for t in acp['acp_cps_per_table'])
    run('replay',[sys.executable,EXP/'scripts/replay_rapid.py','--data',DATA,'--rapid',previous,'--out',OUT/'traffic'])
    save(OUT/'complete.json',{'success':True,'source':'current reproduction/working + reproduction/scripts snapshot',
       'recipe':'latest documented full pipeline, PARALLEL_RAPID_RUNBOOK; same fixed owner-chain parameters, no result-based tuning',
       'final_case':str(previous),'all_tables_unique':True})
if __name__=='__main__':main()
