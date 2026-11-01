"""Figure 6(a) using the code's actual Top-K hot IDs, separately from frequency bins."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_case import LENGTHS
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True)
a=p.parse_args();case=Path(a.case);cfg=json.loads((case/'config.json').read_text())
G=cfg['ranks'];B=8192;NB=cfg['batches'];lengths=cfg['table_lengths'];dense=cfg['dense']
width=1+dense+sum(lengths);offsets=np.cumsum([0]+lengths)
data=np.memmap(cfg['input'],dtype='<u4',mode='r').reshape(-1,width)
hot=[]
for t in range(len(lengths)):
    f=case/'hot'/f'{t}.txt'
    hot.append(np.array(sorted(int(line.split(':')[0].strip()) for line in f.read_text().splitlines() if line.strip()),dtype=np.uint32) if f.exists() else np.array([],dtype=np.uint32))
counts=np.zeros((2,2),dtype=np.int64)
for b in range(NB):
    for r in range(G):
        rows=data[(b*G+r)*B:(b*G+r+1)*B,1+dense:]
        for t in range(len(lengths)):
            ids=rows[:,offsets[t]:offsets[t+1]].ravel();remote=ids[ids%G!=r]
            for kind,values in [(0,remote),(1,np.unique(remote))]:
                if len(hot[t]):
                    pos=np.searchsorted(hot[t],values);valid=pos<len(hot[t]);pos=np.minimum(pos,len(hot[t])-1)
                    nhot=np.count_nonzero(valid&(hot[t][pos]==values))
                else:nhot=0
                counts[kind,0]+=nhot;counts[kind,1]+=len(values)-nhot
    print('traffic split batch',b+1,flush=True)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(case=str(case.resolve()),hot_definition='actual code Top-K frequency list',
    topk=sum(map(len,hot)),rows=NB*G*B,ranks=G,batches=NB,
    traffic_scope='Hash remote requests only; per-rank batch deduplication for Unique',
    counts_raw_then_unique_hot_then_cold=counts.tolist(),
    hot_share_raw_then_unique=(counts[:,0]/counts.sum(axis=1)).tolist(),
    cold_share_raw_then_unique=(counts[:,1]/counts.sum(axis=1)).tolist()),indent=2))
