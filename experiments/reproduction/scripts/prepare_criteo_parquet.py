"""Official Criteo Parquet -> original code's shuffled multi-hot format.

Every categorical string is interpreted as hex, then modulo the original
code's NUM_EMBEDDINGS. Lookup PRNG and shuffle match the original code.
An explicit file manifest distinguishes pilot subsets from full-day runs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import multiprocessing as mp
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('original',ROOT/'reproduction/baseline/Rapid/data_preparation/criteo_singlehot_to_multihot.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
TABLES=None

def convert_part(job):
    import pyarrow.parquet as pq
    filename,part,dense_min=job
    dense_cols=[f'integer_feature_{i}' for i in range(1,14)]
    sparse_cols=[f'categorical_feature_{i}' for i in range(1,27)]
    offsets=original.build_sparse_offsets()
    pf=pq.ParquetFile(filename)
    names=pf.schema_arrow.names
    label=[x for x in names if x not in dense_cols+sparse_cols]
    if len(label)!=1:raise ValueError(f'Unknown schema: {names}')
    n=0
    with open(part,'wb') as dest:
        for batch in pf.iter_batches(batch_size=100000,columns=label+dense_cols+sparse_cols,use_threads=False):
            data=batch.to_pydict();rec=np.empty(batch.num_rows,dtype=original.RECORD_DTYPE)
            rec['label']=data[label[0]]
            dense=np.column_stack([[v if v is not None else 0 for v in data[c]] for c in dense_cols]).astype(np.int32)
            rec['dense']=original.transform_dense(dense,dense_min)
            for t,c in enumerate(sparse_cols):
                ids=np.fromiter((0 if v is None or v=='' else int(v,16)%original.NUM_EMBEDDINGS[t] for v in data[c]),dtype=np.int32,count=batch.num_rows)
                start,end=offsets[t];rec['sparse'][:,start:end]=TABLES[t][ids]
            rec.tofile(dest);n+=len(rec)
    return part,n

def lookup_tables(cache):
    cache.mkdir(parents=True,exist_ok=True)
    ready=cache/'complete.json'
    if not ready.exists():
        rng=np.random.RandomState(0)
        for t,(n,h) in enumerate(zip(original.NUM_EMBEDDINGS,original.MULTI_HOT_SIZES)):
            path=cache/f'table_{t}.npy'
            table=np.lib.format.open_memmap(path,mode='w+',dtype=np.int32,shape=(n,h))
            for start in range(0,n,100000):
                end=min(n,start+100000)
                table[start:end,0]=np.arange(start,end,dtype=np.int32)
                table[start:end,1:]=rng.randint(0,n,size=(end-start,h-1))
            table.flush();del table
            print('lookup',t,n,h,flush=True)
        ready.write_text(json.dumps(dict(seed=0,sizes=original.MULTI_HOT_SIZES,embeddings=original.NUM_EMBEDDINGS)))
    else:
        meta=json.loads(ready.read_text())
        assert meta==dict(seed=0,sizes=original.MULTI_HOT_SIZES,embeddings=original.NUM_EMBEDDINGS)
    # The machine has >400 GiB available RAM. Eager sequential reads avoid
    # random mmap faults while expanding high-cardinality tables.
    return [np.load(cache/f'table_{t}.npy') for t in range(26)]

def main():
    global TABLES
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser()
    p.add_argument('--files',nargs='+',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--cache',default=str(ROOT/'data/multihot_lookup_seed0'))
    p.add_argument('--scope',required=True)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    dense_cols=[f'integer_feature_{i}' for i in range(1,14)]
    sparse_cols=[f'categorical_feature_{i}' for i in range(1,27)]
    dense_min=None;rows=0
    for f in a.files:
        pf=pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=100000,columns=dense_cols):
            dense=np.column_stack([np.asarray(batch.column(i).fill_null(0)) for i in range(13)]).astype(np.int32)
            m=int(dense.min());dense_min=m if dense_min is None else min(m,dense_min)
            rows+=len(dense)
    if not rows:raise ValueError('empty input')
    TABLES=lookup_tables(Path(a.cache))
    temp=out.with_suffix('.unshuffled.bin');n=0
    jobs=[(f,str(out)+f'.part{i}',dense_min) for i,f in enumerate(a.files)]
    # Fork shares read-only lookup pages; imap preserves source file order.
    with mp.get_context('fork').Pool(a.workers) as pool, temp.open('wb') as dest:
        for part,count in pool.imap(convert_part,jobs,chunksize=1):
            with open(part,'rb') as src:shutil.copyfileobj(src,dest,16<<20)
            Path(part).unlink();n+=count
            print('converted',n,'/',rows,flush=True)
    TABLES=None
    assert n==rows and temp.stat().st_size==rows*912
    perm=np.random.RandomState(42).permutation(rows)
    # Full day fits in available RAM; avoids random disk reads during shuffle.
    source=np.fromfile(temp,dtype=original.RECORD_DTYPE)
    with out.open('wb') as dest:
        for start in range(0,rows,100000):source[perm[start:start+100000]].tofile(dest)
    del source
    temp.unlink()
    meta=dict(scope=a.scope,files=a.files,rows=rows,dense_min=dense_min,record_bytes=912,
              lookup_seed=0,shuffle_seed=42,table_lengths=original.MULTI_HOT_SIZES,
              embeddings=original.NUM_EMBEDDINGS)
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print('finished',out,rows,flush=True)

if __name__=='__main__':main()
