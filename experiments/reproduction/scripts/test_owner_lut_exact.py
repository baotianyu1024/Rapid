"""Differential coverage for compact lookup, sparse-ID fallback and map archives."""
import os,subprocess,tempfile,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(dir=root) as tmp:
 p=Path(tmp);case=p/'case';(case/'scheduled').mkdir(parents=True)
 cfg=dict(ranks=4,batches=3,batch_size=8,dense=1,table_lengths=[2,1]);(case/'config.json').write_text(json.dumps(cfg))
 rng=np.random.RandomState(17)
 for r in range(4):
  a=np.zeros((24,5),dtype='<u4');a[:,0]=np.arange(24)+24*r;a[:,2:4]=rng.choice([0,7,19,30],size=(24,2));a[:,4]=rng.choice([0,10000000,30000000],size=24);a.tofile(case/'scheduled'/f'dataset_{r}.bin')
 env=dict(os.environ,PYTHONPATH=str(root/'scripts'),OPENBLAS_NUM_THREADS='1')
 scripts=[root/'speed_work/parallel/refine_trace_owners.reference.py',root/'scripts/refine_trace_owners.py',root/'scripts/refine_trace_owners.py']
 for i,script in enumerate(scripts):
  out=p/str(i);cmd=[os.sys.executable,str(script),'--case',str(case),'--output',str(out),'--no-base-counts','--target-rank','3','--target-fraction','.1']
  if i==2:cmd+=['--uncompressed-maps']
  subprocess.run(cmd,env=env,check=True,stdout=subprocess.DEVNULL)
  if i:
   for r in range(4):assert (p/'0'/f'scheduled/dataset_{r}.bin').read_bytes()==(out/f'scheduled/dataset_{r}.bin').read_bytes()
   for t in range(2):
    with np.load(p/'0'/f'table_{t}_remap.npz') as a,np.load(out/f'table_{t}_remap.npz') as b:
     for k in a.files:assert np.array_equal(a[k],b[k])
 print('PASS compact LUT, sparse-range fallback, compressed/uncompressed exact output and map parity')
