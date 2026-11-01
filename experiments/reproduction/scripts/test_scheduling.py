"""Check sample conservation and owner-preserving remapping across batches."""
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

root=Path(__file__).resolve().parents[2]
binary=root/'reproduction/working/cps/generate_local_dataset'
with tempfile.TemporaryDirectory(dir=root/'reproduction') as tmp:
    tmp=Path(tmp);placement=tmp/'placement';placement.mkdir()
    ranks=4;batch=8;steps=3;n=ranks*batch*steps
    rows=np.zeros((n,5),dtype='<u4');rows[:,0]=np.arange(n)
    rng=np.random.RandomState(7);rows[:,2:]=rng.randint(0,24,size=(n,3))
    rows.tofile(tmp/'input.bin')
    for d in range(ranks):
        (placement/f'table_{d}').write_text(''.join(f'{t}:{i}\n' for t in range(2) for i in range(24) if i%ranks==d))
    for mode,rounds,reduction,algorithm in [('sequential',0,False,None),('greedy_repair',0,False,None),
            ('ours',0,False,None),('ours',2,False,None),('ours',2,True,None),('ours',2,False,'hybrid')]:
        out=tmp/f'{mode}_{rounds}_{reduction}_{algorithm or "default"}'
        cmd=[str(binary),'--input_file',str(tmp/'input.bin'),'--table_files_dir',str(placement),
             '--output_dir',str(out),'--num_devices',str(ranks),'--train_batch_size',str(batch),
             '--steps_per_chunk','1','--num_chunks',str(steps),'--num_threads','1',
             '--dense_count','1','--gamma','0.2','--scheduling',mode,'--swap_rounds',str(rounds),
             '--table_lengths','2','1']
        if reduction:cmd+=['--reduction_tables','1']
        if algorithm:cmd+=['--swap_algorithm',algorithm,'--peak_weight','4',
                            '--swap_target_numerator','6','--swap_target_denominator','5',
                            '--swap_stop_at_target','1']
        subprocess.run(cmd,check=True,env=dict(os.environ,OMP_NUM_THREADS='2'),stdout=subprocess.DEVNULL)
        shards=[np.fromfile(out/f'dataset_{d}.bin',dtype='<u4').reshape(-1,5) for d in range(ranks)]
        assert all(len(s)==batch*steps for s in shards)
        mapping={}
        for b in range(steps):
            records=np.concatenate([s[b*batch:(b+1)*batch] for s in shards])
            assert np.array_equal(np.sort(records[:,0]),np.arange(b*ranks*batch,(b+1)*ranks*batch))
            for rec in records:
                original=rows[rec[0]]
                assert rec[1]==original[1]
                for col,t in [(2,0),(3,0),(4,1)]:
                    key=(t,int(original[col]));value=int(rec[col])
                    assert value%ranks==int(original[col])%ranks
                    if key in mapping:assert mapping[key]==value
                    mapping[key]=value
        assert len(set((t,value) for (t,_),value in mapping.items()))==len(mapping)
        if not reduction and not algorithm:
            old=tmp/f'old_{mode}_{rounds}';previous=list(cmd)
            previous[0]=str(binary)+'.pre_acp_objective'
            previous[previous.index('--output_dir')+1]=str(old)
            subprocess.run(previous,check=True,env=dict(os.environ,OMP_NUM_THREADS='2'),stdout=subprocess.DEVNULL)
            for d in range(ranks):assert (old/f'dataset_{d}.bin').read_bytes()==(out/f'dataset_{d}.bin').read_bytes()
        print('PASS',mode,rounds,'Reduction objective',reduction,'algorithm',algorithm or 'default',
              'capacity, per-batch conservation, all-table mapping',
              'default-byte parity' if not algorithm else 'hybrid invariants')
