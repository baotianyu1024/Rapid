"""Check parallel ordering and exact serial parity, including source offsets."""
import json,os,subprocess,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ref=ROOT/'speed_work/parallel/scheduler.reference';fast=ROOT/'speed_work/parallel/scheduler'
env=dict(os.environ,OMP_NUM_THREADS='8')
with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
 p=Path(tmp);(p/'placement').mkdir();ranks,batch,steps=4,64,12
 rows=np.zeros((ranks*batch*(steps+1),8),dtype='<u4');rows[:,0]=np.arange(len(rows));rng=np.random.RandomState(19)
 rows[:,2:]=rng.randint(0,2000,size=(len(rows),6));rows[:ranks*batch,2:]=0
 rows.tofile(p/'input.bin')
 for d in range(ranks):(p/'placement'/f'table_{d}').write_text(''.join(f'{t}:{i}\n' for t in range(2) for i in range(2000) if i%ranks==d))
 cases=[('ours','ratio',False),('ours','weighted',False),('ours','hybrid',False),('ours','ratio',True),('sequential','ratio',False),('greedy_repair','ratio',False)]
 checks=0
 for ci,(mode,algo,reduction) in enumerate(cases):
  for offset in [0,ranks*batch]:
   common=['--input_file',str(p/'input.bin'),'--table_files_dir',str(p/'placement'),'--num_devices',str(ranks),'--train_batch_size',str(batch),'--num_chunks',str(steps),'--steps_per_chunk','1','--dense_count','1','--table_lengths','3','3','--scheduling',mode,'--swap_algorithm',algo,'--swap_rounds','2','--swap_screen_queries_per_gpu','50000','--swap_evaluations_per_gpu','1000','--start_sample_idx',str(offset)]
   if algo=='weighted':
    ix=common.index('--swap_screen_queries_per_gpu');del common[ix:ix+2]
   if reduction:common+=['--reduction_tables','1']
   outputs=[]
   for binary,workers in [(ref,1),(fast,1),(fast,2),(fast,8)]:
    out=p/f'{ci}_{offset}_{binary.name}_{workers}'
    log=subprocess.check_output([str(binary),*common,'--num_threads',str(workers),'--output_dir',str(out)],env=env)
    outputs.append([(out/f'dataset_{d}.bin').read_bytes() for d in range(ranks)])
   assert all(x==outputs[0] for x in outputs[1:]),(mode,algo,reduction,offset)
   shards=[np.frombuffer(x,dtype='<u4').reshape(-1,8) for x in outputs[-1]]
   for b in range(steps):
    labels=np.concatenate([s[b*batch:(b+1)*batch,0] for s in shards]);assert np.array_equal(np.sort(labels),np.arange(offset+b*batch*ranks,offset+(b+1)*batch*ranks))
   checks+=3;print('PASS',mode,algo,'reduction',reduction,'offset',offset,flush=True)
 print('PASS',checks,'exact serial/parallel comparisons, per-batch sample conservation and input offsets')
