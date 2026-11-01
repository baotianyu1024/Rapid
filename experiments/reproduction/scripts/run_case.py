"""Run an auditable CPU trace experiment on one real-data configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import shutil
from measure import measure
import numpy as np

LENGTHS=[3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--dataset',choices=['taobao','criteo','kaggle'],required=True)
    p.add_argument('--repo',required=True)
    p.add_argument('--ranks',type=int,required=True)
    p.add_argument('--batches',type=int,default=4)
    p.add_argument('--output',required=True)
    p.add_argument('--swap',choices=['default','off','ratio'],default='default')
    p.add_argument('--strict-capacity',action='store_true')
    p.add_argument('--seed-assignment')
    p.add_argument('--base-case')
    p.add_argument('--hot-batches',type=int,default=80)
    a=p.parse_args();out=Path(a.output).resolve();repo=Path(a.repo).resolve();inp=Path(a.input).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    dense=1 if a.dataset=='taobao' else 13
    lengths=[1]*4 if a.dataset=='taobao' else LENGTHS
    n=a.ranks*8192*a.batches
    if inp.stat().st_size < n*4*(1+dense+sum(lengths)):raise ValueError('Not enough complete batches')
    config=vars(a)|dict(input=str(inp),repo=str(repo),rows=n,batch_size=8192,dense=dense,
                       table_lengths=lengths,topk=40000,scope='prefix of complete batches of shuffled input',
                       gpu_execution=False)
    if a.seed_assignment:
        config['seed_assignment_sha256']=hashlib.sha256(Path(a.seed_assignment).read_bytes()).hexdigest()
    config['input_total_rows']=inp.stat().st_size//(4*(1+dense+sum(lengths)))
    if config['input_total_rows']//(a.ranks*8192)==a.batches:
        config['scope']='all complete global batches of supplied input'
    if inp.with_suffix('.json').exists():
        config['input_metadata']=json.loads(inp.with_suffix('.json').read_text())
    hot_rows=min(a.batches,a.hot_batches)*a.ranks*8192
    config['hot_rows']=hot_rows
    (out/'config.json').write_text(json.dumps(config,indent=2))
    provenance={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in repo.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and (p.suffix in ['.py','.cpp','.h','.sh'] or p.name in ['arrange_ids_to_devices','generate_local_dataset','verify_hit_rate'])}
    (out/'code_sha256.json').write_text(json.dumps(provenance,indent=2))
    def run(name,cmd):
        cmd=list(map(str,cmd));start=time.time()
        (out/f'{name}.command.json').write_text(json.dumps(cmd,indent=2))
        with (out/f'{name}.log').open('w') as log:
            r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='16'))
        (out/f'{name}.status.json').write_text(json.dumps(dict(exit_code=r.returncode,seconds=time.time()-start)))
        if r.returncode:raise RuntimeError(f'{name} failed; see {out/name}.log')
    common=['--table_lengths',*lengths,'--dense_count',dense]
    if a.base_case:
        base=Path(a.base_case).resolve();base_cfg=json.loads((base/'config.json').read_text())
        for key,value in [('input',str(inp)),('ranks',a.ranks),('rows',n),('dense',dense),('table_lengths',lengths)]:
            if base_cfg.get(key) != value:raise ValueError(f'Base case mismatch for {key}')
        if base_cfg.get('hot_rows') != hot_rows:raise ValueError('Base case hot-profile scope mismatch')
        shutil.copytree(base/'hot',out/'hot')
        (out/'hot.status.json').write_text(json.dumps(dict(exit_code=0,seconds=0,reused_from=str(base))))
    else:
        run('hot',[sys.executable,repo/'cps/hot_id_extraction.py','--input',inp,'--output',out/'hot',
                   '--topk',40000,'--max_rows',hot_rows,*common])
    placement_extra=['--strict_export_capacity',1] if a.strict_capacity else []
    if a.seed_assignment:placement_extra += ['--seed_assignment',a.seed_assignment]
    run('placement',[repo/'cps/arrange_ids_to_devices','--input_bin',inp,'--hot_ids_dir',out/'hot',
              '--output_dir',out/'placement','--batch_size',8192,'--num_devices',a.ranks,'--n_batches',a.batches,*placement_extra,*common])
    extra=[]
    if a.swap=='off':extra=['--swap_rounds',0]
    if a.swap=='ratio':extra=['--swap_algorithm','ratio']
    run('schedule',[repo/'cps/generate_local_dataset','--input_file',inp,'--table_files_dir',out/'placement',
              '--output_dir',out/'scheduled','--num_devices',a.ranks,'--train_batch_size',8192,
              '--num_chunks',a.batches,'--steps_per_chunk',1,'--num_threads',1,'--gamma',0.2,*extra,*common])
    run('legacy_verify',[repo/'analysis/verify_hit_rate','--dataset_dir',out/'scheduled',
              '--num_devices',a.ranks,'--batch_size',8192,'--num_batches',a.batches,'--no_shuffle',*common])
    for tag,paths,cps in [('hash',[inp],False),('cps',[out/'scheduled'/f'dataset_{r}.bin' for r in range(a.ranks)],True)]:
        result,raw=measure(paths,a.ranks,8192,a.batches,lengths,dense,cps)
        (out/f'{tag}_metrics.json').write_text(json.dumps(result,indent=2))
        np.savez_compressed(out/f'{tag}_counts.npz',**raw)
    (out/'complete.json').write_text(json.dumps(dict(completed=time.time())))
    print('case complete',out,flush=True)

if __name__=='__main__':main()
