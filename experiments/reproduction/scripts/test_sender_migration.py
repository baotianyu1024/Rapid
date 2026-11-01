"""Independent hand trace: move owners without replication or extra traffic."""
import json
from pathlib import Path
import subprocess,sys,tempfile
import numpy as np
from measure import measure
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp);case=tmp/'case';(case/'scheduled').mkdir(parents=True);paths=[]
 for r,ids in enumerate([[0,3,6],[0,3,6,1],[0,3,6,2]]):
  rows=np.column_stack([np.arange(r*8192,(r+1)*8192),np.resize(ids,8192)]).astype('<u4');path=case/'scheduled'/f'dataset_{r}.bin';rows.tofile(path);paths.append(path)
 cfg=dict(ranks=3,batches=1,batch_size=8192,table_lengths=[1],dense=0,dataset='unit_test',rows=24576)
 (case/'config.json').write_text(json.dumps(cfg));metrics,raw=measure(paths,3,8192,1,[1],0,True)
 assert raw['send'].tolist()==[[[6,0,0]]] and raw['receive'].tolist()==[[[0,3,3]]]
 np.savez_compressed(case/'cps_counts.npz',**raw);(case/'hash_metrics.json').write_text(json.dumps(metrics));out=tmp/'migrated'
 subprocess.run([sys.executable,str(Path(__file__).with_name('migrate_senders.py')),'--case',str(case),'--output',str(out),'--rounds','10'],check=True,stdout=subprocess.DEVNULL)
 actual=np.load(out/'cps_counts.npz');assert actual['send'].tolist()==[[[2,2,2]]] and actual['receive'].tolist()==[[[2,2,2]]]
 remap=np.load(out/'table_0_remap.npz');inverse=dict(zip(map(int,remap['new']),map(int,remap['old'])))
 for r in range(3):
  source=np.fromfile(paths[r],dtype='<u4').reshape(-1,2);dest=np.fromfile(out/'scheduled'/f'dataset_{r}.bin',dtype='<u4').reshape(-1,2)
  dest[:,1]=[inverse.get(int(x),int(x)) for x in dest[:,1]]
  assert np.array_equal(source,dest)
print('PASS: sender [6,0,0]->[2,2,2], receiver [0,3,3]->[2,2,2], total traffic and exact inverse-mapped records preserved')
