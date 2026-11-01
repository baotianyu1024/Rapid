"""Isolate archive compression from actual record remapping; byte-compare output datasets."""
import json,time,os,hashlib
from pathlib import Path
import numpy as np
from refine_balanced_owners import materialize
root=Path(__file__).resolve().parents[1];case=root/'results/criteo_day6_32_owner_balanced_16b';seed=root/'results/criteo_day6_32_price_warm';cache=root/'balanced_owner_cache32';out=root/'results/speed_exact';out.mkdir(exist_ok=True)
cfg=json.loads((case/'config.json').read_text());tables=[dict(ids=np.load(cache/f'{t}.ids.npy',mmap_mode='r'),owners=np.load(seed/f'{t}.owners.npy')) for t in range(26)];counts=np.load(seed/'predicted_counts.npz');rows=[]
for compressed in [True,False]:
 target=out/('materialize_compressed' if compressed else 'materialize_stored');target.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time()
 materialize(case,cache,target,cfg,tables,counts['receive'],counts['send'],compressed=compressed)
 row=dict(compressed=compressed,external_wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,phases=json.loads((target/'materialization.json').read_text()),map_archive_bytes=sum(p.stat().st_size for p in target.glob('table_*_remap.npz')))
 for r in range(32):
  expected=seed/'scheduled'/f'dataset_{r}.bin';actual=target/'scheduled'/f'dataset_{r}.bin'
  def digest(p):
   h=hashlib.sha256()
   with p.open('rb') as f:
    while True:
     buf=f.read(8*1024*1024)
     if not buf:break
     h.update(buf)
   return h.hexdigest()
  assert digest(expected)==digest(actual)
 for t in range(26):
  with np.load(target/f'table_{t}_remap.npz') as actual,np.load(seed/f'table_{t}_remap.npz') as expected:
   for k in expected.files:assert np.array_equal(actual[k],expected[k])
 row['all_mapping_arrays_and_dataset_bytes_equal']=True;rows.append(row)
 (out/'materialization_benchmark.json').write_text(json.dumps(rows,indent=2));print('BENCHMARK',row,flush=True)
