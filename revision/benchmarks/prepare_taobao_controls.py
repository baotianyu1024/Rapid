"""Construct offline HugeCTR controls from the fresh Taobao sample."""
from pathlib import Path
import json
import shutil
import numpy as np

case = Path('/tmp/rapid_fresh_stage/taobao_8_v2')
out = Path('/tmp/rapid_fresh_stage/taobao_controls')
if shutil.disk_usage(out.parent).free < 1024**3:
    raise RuntimeError('Need at least 1 GiB for controls and reserve')
out.mkdir(exist_ok=True)
meta = json.loads((case / 'data/input.json').read_text())
rows, offsets = meta['rows'], meta['active_id_offsets']
raw = np.memmap(case / 'data/input.bin', dtype='<u4', mode='r').reshape(-1, 6)
sparse = np.load(case / 'data/sparse.npy', mmap_mode='r')
with (out / 'native_compact.bin').open('wb') as f:
    for first in range(0, rows, 8192):
        batch = raw[first:first+8192].copy()
        batch[:, 2:] = sparse[first:first+8192] - np.asarray(offsets[:-1], dtype=np.uint32)
        batch.tofile(f)
shards = [np.memmap(case / 'optimized/peak_samples/scheduled' / f'dataset_{rank}.bin',
                    dtype='<u4', mode='r').reshape(-1, 6) for rank in range(8)]
with (out / 'materialized.bin').open('wb') as f:
    for batch in range(rows // 65536):
        for shard in shards:
            shard[batch*8192:(batch+1)*8192].tofile(f)
print('native sizes:', [b-a for a, b in zip(offsets, offsets[1:])])
