"""Hand-built asymmetric traffic: same-table swaps balance [4,1,1] -> [2,2,2]."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from measure import measure

with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp);case=tmp/'case';(case/'scheduled').mkdir(parents=True);paths=[]
    for r,ids in enumerate([[0,3,1,2],[0,3,1],[0,3,2]]):
        rows=np.column_stack([np.arange(r*8192,(r+1)*8192),np.resize(ids,8192)]).astype('<u4')
        path=case/'scheduled'/f'dataset_{r}.bin';rows.tofile(path);paths.append(path)
    cfg=dict(ranks=3,batches=1,batch_size=8192,table_lengths=[1],dense=0,dataset='unit_test',rows=24576)
    (case/'config.json').write_text(json.dumps(cfg))
    metrics,raw=measure(paths,3,8192,1,[1],0,True)
    assert raw['send'].tolist()==[[[4,1,1]]]
    np.savez_compressed(case/'cps_counts.npz',**raw)
    (case/'hash_metrics.json').write_text(json.dumps(metrics))
    out=tmp/'refined'
    subprocess.run([sys.executable,str(Path(__file__).with_name('balance_senders.py')),
                    '--case',str(case),'--output',str(out),'--rounds','10'],check=True,stdout=subprocess.DEVNULL)
    result=np.load(out/'cps_counts.npz');assert result['send'].tolist()==[[[2,2,2]]]
    assert np.array_equal(result['receive'],raw['receive'])
    for r in range(3):
        original=np.fromfile(paths[r],dtype='<u4').reshape(-1,2)
        transformed=np.fromfile(out/'scheduled'/f'dataset_{r}.bin',dtype='<u4').reshape(-1,2)
        assert np.array_equal(original[:,0],transformed[:,0])
print('PASS sender [4,1,1] -> [2,2,2]; receive counts and samples unchanged')
