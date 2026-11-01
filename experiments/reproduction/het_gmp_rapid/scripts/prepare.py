"""Lossless table-disjoint active-ID compaction for the original HET partitioner."""
import argparse, hashlib, time
import numpy as np
from common import *

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--out',required=True); p.add_argument('--batches',type=int,default=64)
p.add_argument('--ranks',type=int,default=P); p.add_argument('--batch-size',type=int,default=B)
p.add_argument('--table-lengths',type=int,nargs='+',default=LENGTHS)
p.add_argument('--dense',type=int,default=DENSE);p.add_argument('--dataset',default='criteo_day6')
a=p.parse_args(); P=a.ranks; B=a.batch_size; LENGTHS=a.table_lengths; DENSE=a.dense
assert DENSE>=0 and all(x>0 for x in LENGTHS)
WIDTH=1+DENSE+sum(LENGTHS)
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
n=a.batches*P*B
assert P > 0 and B > 0 and a.batches > 0
assert n*sum(LENGTHS) < np.iinfo(np.int32).max, 'Original HET-GMP edge offsets overflow int32'
inp=Path(a.input); assert inp.stat().st_size >= n*WIDTH*4
digest=hashlib.sha256(); t=time.perf_counter()
with inp.open('rb') as f, (out/'input.bin').open('wb') as g:
    remain=n*WIDTH*4
    while remain:
        b=f.read(min(remain,64*1024**2)); assert b
        g.write(b); digest.update(b); remain-=len(b)
copy_seconds=time.perf_counter()-t
raw=np.memmap(out/'input.bin',dtype=np.int32,mode='r',shape=(n,WIDTH))
mapped=np.lib.format.open_memmap(out/'sparse.npy',mode='w+',dtype=np.int32,shape=(n,sum(LENGTHS)))
offsets=[0]; col=0; tables=[]; start=time.perf_counter()
for ti,l in enumerate(LENGTHS):
    t=time.perf_counter()
    ids=raw[:,1+DENSE+col:1+DENSE+col+l]
    assert ids.min()>=0
    # Histogram + rank lookup preserves the same ascending dense mapping while
    # avoiding sorting hundreds of millions of occurrences for multi-hot tables.
    frequency=np.bincount(ids.ravel())
    unique=np.flatnonzero(frequency).astype(np.int32)
    lookup=np.cumsum(frequency>0,dtype=np.int32)-1
    for row in range(0,n,65536):
        mapped[row:row+65536,col:col+l]=lookup[ids[row:row+65536]]+offsets[-1]
    assert np.array_equal(unique[lookup[ids[:1000]]],ids[:1000])
    np.save(out/f'ids_{ti}.npy',unique)
    offsets.append(offsets[-1]+len(unique)); col+=l
    tables.append(dict(table=ti,unique_ids=len(unique),maximum_degree=int(frequency.max()),seconds=time.perf_counter()-t))
    print('compact',tables[-1],flush=True)
    del ids,unique,lookup,frequency
mapped.flush()
save(out/'input.json',dict(dataset=a.dataset,input_source=str(inp.resolve()),input_sha256=digest.hexdigest(),rows=n,global_batches=a.batches,ranks=P,local_batch=B,lengths=LENGTHS,dense=DENSE,dim=DIM,record_bytes=WIDTH*4,scope='prefix of existing shuffled '+a.dataset+'; not full dataset',active_id_offsets=offsets,active_embeddings=offsets[-1],copy_seconds=copy_seconds,het_adapter_seconds=time.perf_counter()-start,tables=tables))
