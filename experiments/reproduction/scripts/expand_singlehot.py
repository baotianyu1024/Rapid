"""Expand parsed Criteo with cached original lookup, then exact seed-42 shuffle."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_criteo_parquet import lookup_tables,original
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
a=p.parse_args();inp=Path(a.input);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():raise FileExistsError(out)
meta=json.loads(Path(str(inp)+'.json').read_text());n=meta['rows'];minimum=meta['dense_min']
assert inp.stat().st_size==n*160
data=np.memmap(inp,dtype='<i4',mode='r',shape=(n,40))
tables=lookup_tables(Path('data/multihot_lookup_seed0'));offsets=original.build_sparse_offsets()
tmp=out.with_suffix('.unshuffled.bin')
with tmp.open('wb') as dest:
    for start in range(0,n,100000):
        batch=data[start:start+100000];rec=np.empty(len(batch),dtype=original.RECORD_DTYPE)
        rec['label']=batch[:,0];rec['dense']=original.transform_dense(batch[:,1:14].copy(),minimum)
        rec['sparse']=original.expand_sparse_to_multi_hot(batch[:,14:],tables,offsets)
        rec.tofile(dest)
        if start%1000000==0:print('expanded',start,'/',n,flush=True)
del tables
source=np.fromfile(tmp,dtype=original.RECORD_DTYPE);assert len(source)==n
permutation=np.random.RandomState(42).permutation(n)
with out.open('wb') as dest:
    for start in range(0,n,100000):source[permutation[start:start+100000]].tofile(dest)
tmp.unlink();meta.update(record_bytes=912,shuffle_seed=42,lookup_seed=0,scope='all Kaggle train.txt rows')
out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('finished',out,n,flush=True)
