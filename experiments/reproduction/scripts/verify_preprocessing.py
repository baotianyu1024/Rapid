"""Verify shuffled records against independently selected raw rows and original transform."""
import argparse
import bisect
import json
from pathlib import Path
import numpy as np
from prepare_criteo_parquet import original
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['kaggle','criteo'],required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
output=root/('data/processed/kaggle/multihot.bin' if a.dataset=='kaggle' else 'data/processed/criteo/day6.bin')
meta=json.loads(output.with_suffix('.json').read_text());n=meta['rows'];data=np.memmap(output,dtype=original.RECORD_DTYPE,mode='r')
assert len(data)==n and output.stat().st_size==n*912
indices=np.unique(np.r_[0,1,2,n//2,n-2,n-1,np.random.default_rng(99).integers(0,n,10)])
perm=np.random.RandomState(42).permutation(n);source_positions=perm[indices].copy();del perm
lookups=[np.load(root/'data/multihot_lookup_seed0'/f'table_{t}.npy',mmap_mode='r') for t in range(26)]
offsets=original.build_sparse_offsets();checks=[]
if a.dataset=='kaggle':single=np.memmap(root/'data/processed/kaggle/singlehot.bin',dtype='<i4',mode='r').reshape(-1,40)
else:
 import pyarrow.parquet as pq
 files=meta['files'];ends=np.cumsum([pq.ParquetFile(f).metadata.num_rows for f in files]);dense_names=[f'integer_feature_{i}' for i in range(1,14)];sparse_names=[f'categorical_feature_{i}' for i in range(1,27)]
for out_idx,pos in zip(indices,source_positions):
 if a.dataset=='kaggle':
  row=single[pos];label=row[0];dense=row[1:14].copy();sparse=row[14:]
 else:
  j=bisect.bisect_right(ends,int(pos));local=int(pos)-(int(ends[j-1]) if j else 0);pf=pq.ParquetFile(files[j]);group=0
  while local>=pf.metadata.row_group(group).num_rows:local-=pf.metadata.row_group(group).num_rows;group+=1
  row=pf.read_row_group(group,use_threads=False).slice(local,1).to_pylist()[0]
  label_name=next(k for k in row if k not in dense_names+sparse_names)
  label=row[label_name];dense=np.array([row[k] or 0 for k in dense_names],dtype=np.int32)
  sparse=np.array([0 if not row[k] else int(row[k],16)%original.NUM_EMBEDDINGS[t] for t,k in enumerate(sparse_names)])
 expected=np.empty(1,dtype=original.RECORD_DTYPE);expected['label']=label;expected['dense']=original.transform_dense(dense.reshape(1,-1),meta['dense_min'])
 for t,(start,end) in enumerate(offsets):expected['sparse'][0,start:end]=lookups[t][sparse[t]]
 assert data[out_idx].tobytes()==expected.tobytes(),(a.dataset,int(out_idx),int(pos))
 checks.append(dict(output_row=int(out_idx),source_row=int(pos),exact_record_equal=True))
 print('verified',a.dataset,int(out_idx),flush=True)
(root/'reproduction'/f'{a.dataset}_preprocessing_record_validation.json').write_text(json.dumps(dict(rows=n,record_bytes=912,all_checks_pass=True,checks=checks),indent=2))
