"""Sequential fresh preprocessing of the selected RAPID configuration and HET-GMP."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import measured, save, LENGTHS

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'reproduction/het_gmp_rapid'
OUT=EXP/'results/timing32_rerun'
SCRIPTS=ROOT/'reproduction/scripts'
WORK=ROOT/'reproduction/working'
OLD=ROOT/'reproduction/results'
DATA=OUT/'data'
PY=sys.executable

def run(name,cmd):
    return measured(name,cmd,OUT,threads=16)

def replace_args(cmd, replacements):
    cmd=list(cmd)
    for flag,value in replacements.items():cmd[cmd.index(flag)+1]=str(value)
    return cmd

def main():
    assert not (OUT/'gmp_partition.command.json').exists(), 'Use a new output folder for another run'
    assert json.loads((OUT/'cleanup_manifest.json').read_text())['completed']
    check=json.loads((OUT/'owner_policy_check.json').read_text())
    prior=json.loads((ROOT/'reproduction/analysis_owner_oracle_criteo32_balanced.json').read_text())
    assert check['receive_matrix']==prior['receive_matrix']
    assert check['oracle_observed_storage']==prior['oracle_observed_storage']
    binary=WORK/'cps/generate_local_dataset.pre_hybrid'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()=='07ca21c6b536162c9d5ef78d51b6835bb53d8fbcc1725e42afac9946cc21b1b5'
    envdir=ROOT/'reproduction/toolchain/lib'
    os.environ['LD_LIBRARY_PATH']=str(envdir)+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    save(OUT/'benchmark_config.json',dict(rows=4194304,ranks=32,local_batch=8192,batches=16,
        rapid='previously selected owner_balanced configuration, fixed parameters; no new search',
        owner_ties=dict(target_rank=20,target_fraction=.1,target_rank_2=6,target_fraction_2=.05),
        gmp_rounds=10,gmp_store_rate=.01,gmp_topology='uniform off-diagonal 0.1',
        order=['GMP','RAPID'],omp_threads=16,openblas_threads=1,cache_policy='normal filesystem cache; not a cold-cache benchmark',
        cleanup_manifest='cleanup_manifest.json',excludes='installation, compilation, prior tuning/search',
        notes='GMP output includes optional label+dense+sparse materialization; RAPID includes optimized owner remapping and ACP profile'))
    # Preparation produces one fresh common input slice and GMP-specific reversible ID encoding.
    run('input_and_gmp_adapter',[PY,EXP/'scripts/prepare.py','--input',ROOT/'data/processed/criteo/day6.bin',
        '--out',DATA,'--batches',16,'--ranks',32])
    assert json.loads((DATA/'input.json').read_text())['input_sha256']=='44ba74ba68f60c13d36d3724cbb7dd95b30d997fc9d489cbd67347dc3fe70317'
    run('gmp_partition',[PY,EXP/'scripts/partition_het.py','--data',DATA,'--out',OUT/'gmp','--ranks',32])
    run('gmp_materialize',[PY,EXP/'scripts/materialize_het.py','--data',DATA,'--partition',OUT/'gmp','--ranks',32])
    model=OUT/'rapid_base'; model.mkdir()
    cfg=json.loads((OLD/'criteo_day6_32_metis_ratio110_16b/config.json').read_text())
    cfg.update(input=str(DATA/'input.bin'),output=str(model),base_case=None,seed_assignment=str(OUT/'rapid_seed/seed_32.txt'))
    save(model/'config.json',cfg)
    common=['--table_lengths',*LENGTHS,'--dense_count',13]
    run('rapid_hot',[PY,WORK/'cps/hot_id_extraction.py','--input',DATA/'input.bin',
        '--output',model/'hot','--topk',40000,'--max_rows',4194304,*common])
    # Append dependency paths, preserving the benchmark env's NumPy 1.26.4.
    code='import sys,runpy;sys.path.extend(sys.argv[1:4]);script=sys.argv[4];sys.argv=sys.argv[4:];runpy.run_path(script,run_name="__main__")'
    run('rapid_metis_seed',[PY,'-c',code,SCRIPTS,ROOT/'reproduction/deps',
        '/home/longjiu.bty/miniconda3/lib/python3.10/site-packages',SCRIPTS/'build_metis_seed.py',
        '--input',DATA/'input.bin','--hot',model/'hot','--output',OUT/'rapid_seed','--rows',2048,'--ranks',32])
    cmd=json.loads((OLD/'criteo_day6_32_metis_seed_16b/placement.command.json').read_text())
    cmd=replace_args(cmd,{'--input_bin':DATA/'input.bin','--hot_ids_dir':model/'hot',
        '--output_dir':model/'placement','--seed_assignment':OUT/'rapid_seed/seed_32.txt'})
    run('rapid_placement',cmd)
    cmd=json.loads((OLD/'criteo_day6_32_metis_ratio110_16b/schedule.command.json').read_text())
    cmd[0]=str(binary)
    cmd=replace_args(cmd,{'--input_file':DATA/'input.bin','--table_files_dir':model/'placement','--output_dir':model/'scheduled'})
    run('rapid_schedule',cmd)
    run('rapid_owner_and_acp',[PY,SCRIPTS/'refine_trace_owners.py','--case',model,'--output',OUT/'rapid_best',
        '--target-rank',20,'--target-fraction',.1,'--target-rank-2',6,'--target-fraction-2',.05,'--no-base-counts'])
    # The final results exist before any optional replay/timing analysis.
    save(OUT/'preprocess_complete.json',dict(completed=time.time(),success=True))
    print('ALL PREPROCESSING COMPLETE',flush=True)

if __name__=='__main__':main()
