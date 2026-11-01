"""Nine feasible schedules, equal data, placement per row, and denominators."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from measure import measure
from run_case import LENGTHS

p=argparse.ArgumentParser()
p.add_argument('--case',required=True,help='completed run_case directory, placement is reused')
p.add_argument('--output',required=True)
a=p.parse_args();case=Path(a.case).resolve();out=Path(a.output).resolve()
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
cfg=json.loads((case/'config.json').read_text())
G=cfg['ranks'];B=cfg['batch_size'];NB=cfg['batches'];dense=cfg['dense'];lengths=cfg['table_lengths']
width=1+dense+sum(lengths);n=G*B*NB
rows=np.memmap(cfg['input'],dtype='<u4',mode='r').reshape(-1,width)[:n,1+dense:]
offsets=np.cumsum([0]+lengths)
keys=[];counts=[]
for t in range(len(lengths)):
    ids,freq=np.unique(rows[:,offsets[t]:offsets[t+1]],return_counts=True)
    keys.append((np.uint64(t)<<np.uint64(32))|ids.astype(np.uint64));counts.append(freq)
keys=np.concatenate(keys);counts=np.concatenate(counts)
freq_order=np.lexsort((keys,-counts))
rr=np.empty(len(keys),dtype=np.int32);rr[freq_order]=np.arange(len(keys))%G
placements={'ours':case/'placement'}
for name,owners in [('hash',(keys%np.uint64(1<<32))%G),('freq_rr',rr)]:
    dest=out/f'placement_{name}';dest.mkdir()
    handles=[(dest/f'table_{r}').open('w') for r in range(G)]
    for key,owner in zip(keys.tolist(),owners.tolist()):handles[owner].write(f'{key>>32}:{key&0xffffffff}\n')
    for f in handles:f.close()
    placements[name]=dest
root=Path(__file__).resolve().parents[2]
binary=root/'reproduction/working/cps/generate_local_dataset'
results={}
for placement,directory in placements.items():
    for schedule in ['sequential','greedy_repair','ours']:
        label=placement+'__'+schedule;dest=out/label
        cmd=[str(binary),'--input_file',cfg['input'],'--table_files_dir',str(directory),
            '--output_dir',str(dest),'--num_devices',str(G),'--train_batch_size',str(B),
            '--num_chunks',str(NB),'--steps_per_chunk','1','--num_threads','1',
            '--dense_count',str(dense),'--gamma','0.2','--scheduling',schedule,'--swap_rounds','0',
            '--table_lengths',*map(str,lengths)]
        (out/f'{label}.command.json').write_text(json.dumps(cmd,indent=2))
        with (out/f'{label}.log').open('w') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='16'))
        metrics,raw=measure([dest/f'dataset_{r}.bin' for r in range(G)],G,B,NB,lengths,dense,True)
        results[label]=metrics
        np.savez_compressed(out/f'{label}.npz',**raw)
        (out/'metrics.json').write_text(json.dumps(results,indent=2))
        print(label,'raw',metrics['local_hit_rate'],'unique',metrics['unique_local_hit_rate'],flush=True)
(out/'config.json').write_text(json.dumps(cfg|dict(source_case=str(case),
    hash='fid % ranks',frequency_ties='ascending table-id then feature-id',
    greedy_repair='initial max-hit lowest rank ties; overflow in original order; best underfilled destination',
    ours='capacity-aware gamma=0.2; swaps disabled',
    locality_denominators=['raw occurrences','unique IDs per rank/batch']),indent=2))
