"""Build offline-only controls from the independently downloaded Kaggle sample."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', choices=('kaggle', 'terabyte'), default='kaggle')
parser.add_argument('--case', type=Path, help='Fresh offline case; defaults to the historical case')
parser.add_argument('--out', type=Path, help='Fresh control directory; defaults to the historical output')
parser.add_argument('--native-only', action='store_true', help='Build hash baseline before CPS scheduling finishes')
parser.add_argument('--materialized-only', action='store_true', help='Add RAPID control to an existing native control directory')
args = parser.parse_args()
if args.native_only and args.materialized_only:
    parser.error('Choose at most one control mode')
root = Path('/tmp/rapid_fresh_stage')
out = args.out or root / ('hugectr_controls' if args.dataset == 'kaggle' else 'hugectr_terabyte_controls')
if out.exists() and any(out.iterdir()) and not args.materialized_only:
    raise FileExistsError(out)
if args.materialized_only and not (out / 'native_compact.bin').is_file():
    raise FileNotFoundError(out / 'native_compact.bin')
out.parent.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage(out.parent).free < 4 * 1024**3:
    raise RuntimeError('Need 4 GiB free for two 912 MiB controls and reserve')
out.mkdir(exist_ok=True)
case = args.case or root / f'{args.dataset}_8_v2'
meta = json.loads((case / 'data/input.json').read_text())
hot = meta['lengths']
offsets = meta['active_id_offsets']
width = 14 + sum(hot)
raw = np.memmap(case / 'data/input.bin', dtype='<u4', mode='r').reshape(-1, width)
sparse = np.load(case / 'data/sparse.npy', mmap_mode='r')
columns = np.cumsum([0] + hot)
if not args.materialized_only:
    with (out / 'native_compact.bin').open('wb') as f:
        for first in range(0, len(raw), 8192):
            rows = raw[first:first+8192].copy()
            ids = sparse[first:first+8192]
            for t in range(len(hot)):
                rows[:, 14+columns[t]:14+columns[t+1]] = ids[:, columns[t]:columns[t+1]] - offsets[t]
            rows.tofile(f)
    (out / 'native_sizes.json').write_text(json.dumps([b-a for a,b in zip(offsets, offsets[1:])]))
if args.native_only:
    print('created native control:', len(raw), 'rows')
    raise SystemExit(0)
shards = [np.memmap(case / 'optimized/peak_samples/scheduled' / f'dataset_{r}.bin',
                    dtype='<u4', mode='r').reshape(-1, width) for r in range(meta['ranks'])]
if shutil.disk_usage(out).free < raw.nbytes + 2 * 1024**3:
    raise RuntimeError('Insufficient space for RAPID materialized control and reserve')
local_batch = meta['local_batch']
global_batch = meta['ranks'] * local_batch
if len(raw) != meta['global_batches'] * global_batch:
    raise ValueError('Offline plan does not cover the full input')
with (out / 'rapid_materialized.bin').open('wb') as f:
    for batch in range(meta['global_batches']):
        for shard in shards:
            shard[batch*local_batch:(batch+1)*local_batch].tofile(f)
print('created controls:', len(raw), 'rows each')
