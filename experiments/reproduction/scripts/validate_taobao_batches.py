"""Exact row-multiset check after reversing the original four-table ID mapping."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--placement',required=True)
p.add_argument('--scheduled',required=True);p.add_argument('--ranks',type=int,default=8);p.add_argument('--batches',type=int,default=4)
p.add_argument('--output',required=True);a=p.parse_args();G=a.ranks;B=8192
entries=[[] for _ in range(4)]
for r in range(G):
    for line in (Path(a.placement)/f'table_{r}').read_text().splitlines():
        t,fid=map(int,line.split(':'));entries[t].append((fid,r))
inverse=[]
for t in range(4):
    counter=np.zeros(G,dtype=np.int64);pairs=[]
    for fid,r in sorted(entries[t]):
        pairs.append((int(counter[r])*G+r,fid));counter[r]+=1
    inv=np.full(max(x[0] for x in pairs)+1,0xffffffff,dtype=np.uint32)
    for newid,oldid in pairs:inv[newid]=oldid
    inverse.append(inv)
raw=np.memmap(a.input,dtype='<u4',mode='r').reshape(-1,6)
shards=[np.memmap(Path(a.scheduled)/f'dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,6) for r in range(G)]
checks=[];decoded_batches=[]
for b in range(a.batches):
    records=np.concatenate([x[b*B:(b+1)*B] for x in shards])
    assert len(records)==G*B
    for t in range(4):
        decoded=inverse[t][records[:,2+t]];assert np.all(decoded!=0xffffffff);records[:,2+t]=decoded
    source=np.ascontiguousarray(raw[b*G*B:(b+1)*G*B]).view('V24').ravel()
    target=np.ascontiguousarray(records).view('V24').ravel()
    decoded_batches.append(target.copy())
    equal=bool(np.array_equal(np.sort(source),np.sort(target)));checks.append(equal)
    print('batch',b,'exact original row multiset preserved',equal)
whole_source=np.ascontiguousarray(raw[:a.batches*G*B]).view('V24').ravel()
whole_target=np.concatenate(decoded_batches)
whole_equal=bool(np.array_equal(np.sort(whole_source),np.sort(whole_target)))
Path(a.output).write_text(json.dumps(dict(check='exact sorted full-record bytes after inverse ID mapping',per_batch_equal=checks,all_equal=all(checks),whole_trace_equal=whole_equal),indent=2))
