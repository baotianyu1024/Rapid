"""Reuse a completed baseline's exact placement and samples for swap ablation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import hashlib
import numpy as np
from measure import measure

p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True)
p.add_argument('--algorithm',choices=['ratio','weighted','hybrid','off'],required=True)
p.add_argument('--screen-budget',type=int)
p.add_argument('--evaluation-budget',type=int)
p.add_argument('--weight',type=int)
p.add_argument('--target-numerator',type=int)
p.add_argument('--target-denominator',type=int)
p.add_argument('--objective-excluded-tables',type=int,nargs='+',default=[])
p.add_argument('--stop-at-target',action='store_true')
p.add_argument('--batches',type=int,help='Run a prefix for parameter screening; defaults to the full base scope')
p.add_argument('--rounds',type=int)
p.add_argument('--source-samples',type=int)
p.add_argument('--candidates',type=int)
p.add_argument('--targets',type=int)
p.add_argument('--cycles',choices=[0,1],type=int)
a=p.parse_args();case=Path(a.case).resolve();out=Path(a.output).resolve()
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True);cfg=json.loads((case/'config.json').read_text());root=Path(__file__).resolve().parents[2]
repo=root/'reproduction/working'
if a.batches is not None:
    if a.batches < 1 or a.batches > cfg['batches']:
        raise ValueError('--batches must be within the completed base case')
    cfg['batches']=a.batches;cfg['rows']=a.batches*cfg['ranks']*cfg['batch_size']
    cfg['scope']='prefix screening run of the completed base case'
cfg.update(repo=str(repo),base_case=str(case),swap=a.algorithm)
cfg.update(screen_budget=a.screen_budget,evaluation_budget=a.evaluation_budget)
cfg['peak_weight']=a.weight
cfg.update(target_numerator=a.target_numerator,target_denominator=a.target_denominator)
cfg['objective_excluded_tables']=a.objective_excluded_tables
cfg['stop_at_target']=a.stop_at_target
cfg.update(swap_rounds=a.rounds,swap_source_samples=a.source_samples,
           swap_candidates=a.candidates,swap_targets=a.targets,swap_cycles=a.cycles)
cfg['objective_note']='Excluded tables retain all output IDs but do not influence Unique scheduling cost; intended for fixed-cost Reduction tables.'
(out/'config.json').write_text(json.dumps(cfg,indent=2))
(out/'code_sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'cps/generate_local_dataset',repo/'cps/generate_local_dataset.cpp',repo/'cps/swap_optimizer.h',Path(__file__),Path(__file__).with_name('measure.py')]},indent=2))
cmd=[str(repo/'cps/generate_local_dataset'),'--input_file',cfg['input'],'--table_files_dir',str(case/'placement'),
    '--output_dir',str(out/'scheduled'),'--num_devices',str(cfg['ranks']),'--train_batch_size','8192',
    '--num_chunks',str(cfg['batches']),'--steps_per_chunk','1','--num_threads','1','--gamma','0.2',
    '--dense_count',str(cfg['dense']),'--table_lengths',*map(str,cfg['table_lengths'])]
cmd+=['--swap_rounds','0'] if a.algorithm=='off' else ['--swap_algorithm',a.algorithm]
if a.screen_budget is not None:cmd+=['--swap_screen_queries_per_gpu',str(a.screen_budget)]
if a.evaluation_budget is not None:cmd+=['--swap_evaluations_per_gpu',str(a.evaluation_budget)]
if a.weight is not None:cmd+=['--peak_weight',str(a.weight)]
if a.target_numerator is not None:cmd+=['--swap_target_numerator',str(a.target_numerator)]
if a.target_denominator is not None:cmd+=['--swap_target_denominator',str(a.target_denominator)]
if a.objective_excluded_tables:cmd+=['--reduction_tables',*map(str,a.objective_excluded_tables)]
if a.stop_at_target:cmd+=['--swap_stop_at_target','1']
for value,flag in [(a.rounds,'--swap_rounds'),(a.source_samples,'--swap_source_samples'),
                   (a.candidates,'--swap_candidates'),(a.targets,'--swap_targets'),
                   (a.cycles,'--swap_cycles')]:
    if value is not None:cmd += [flag,str(value)]
(out/'schedule.command.json').write_text(json.dumps(cmd,indent=2));start=time.time()
with (out/'schedule.log').open('w') as f:run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='16'))
(out/'schedule.status.json').write_text(json.dumps(dict(exit_code=run.returncode,seconds=time.time()-start)))
if run.returncode:raise RuntimeError('Scheduling failed; see schedule.log')
metrics,raw=measure([out/'scheduled'/f'dataset_{r}.bin' for r in range(cfg['ranks'])],cfg['ranks'],8192,cfg['batches'],cfg['table_lengths'],cfg['dense'],True)
(out/'cps_metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(out/'cps_counts.npz',**raw)
shutil.copyfile(case/'hash_metrics.json',out/'hash_metrics.json')
(out/'complete.json').write_text(json.dumps(dict(completed=time.time())))
print('variant complete',out,flush=True)
