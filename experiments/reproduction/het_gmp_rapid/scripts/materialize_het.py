"""Optional like-for-like disk materialization, separate from graph partition time."""
import argparse, time
import numpy as np
from common import *
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--partition',required=True)
p.add_argument('--ranks',type=int,default=P);a=p.parse_args();P=a.ranks
data=Path(a.data);part=Path(a.partition);out=part/'scheduled';out.mkdir(exist_ok=True)
meta=json.loads((data/'input.json').read_text());WIDTH=meta['record_bytes']//4
assert WIDTH==1+meta['dense']+sum(meta['lengths']) and meta['ranks']==P
assignment=np.load(part/'assignment.npy'); raw=np.memmap(data/'input.bin',dtype=np.int32,mode='r').reshape(-1,WIDTH)
counts=[]
for r in range(P):
    idx=np.flatnonzero(assignment==r);counts.append(len(idx))
    with (out/f'dataset_{r}.bin').open('wb') as f:
        for start in range(0,len(idx),65536):raw[idx[start:start+65536]].tofile(f)
assert sum(counts)==len(raw)
save(part/'materialization.json',dict(rows=len(raw),rank_rows=counts,bytes=len(raw)*WIDTH*4,note='Full label+dense+sparse records, original IDs, rank grouping in original row order; input sparse.npy separately feeds replay.'))
