"""Balance the training prefix while keeping HET-GMP embedding placement."""
import argparse
import os
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--train-rows', type=int, required=True)
p.add_argument('--ranks', type=int, default=8)
a = p.parse_args()
if a.out.exists():
    raise FileExistsError(a.out)
with np.load(a.source) as source:
    arrays = {key: source[key] for key in source.files}
data = arrays['data_partition'][:a.train_rows].copy()
original = data.copy()
if len(data) % a.ranks:
    raise ValueError('training rows must divide evenly across ranks')
target = len(data) // a.ranks
counts = np.bincount(data, minlength=a.ranks)
for donor in range(a.ranks):
    excess = int(counts[donor] - target)
    if excess <= 0:
        continue
    candidates = np.flatnonzero(data == donor)[-excess:]
    offset = 0
    for receiver in range(a.ranks):
        count = max(0, min(excess - offset, int(target - counts[receiver])))
        if count:
            data[candidates[offset:offset + count]] = receiver
            counts[receiver] += count
            counts[donor] -= count
            offset += count
        if offset == excess:
            break
    if offset != excess:
        raise RuntimeError('could not balance partition')
if not np.all(np.bincount(data, minlength=a.ranks) == target):
    raise RuntimeError('unbalanced partition')
arrays['data_partition'] = data
a.out.parent.mkdir(parents=True, exist_ok=True)
stat = os.statvfs(a.out.parent)
if stat.f_bavail * stat.f_frsize < a.source.stat().st_size + 2**30:
    raise OSError('insufficient free space for partition')
np.savez_compressed(a.out, **arrays)
print(f'target_rows_per_rank={target} moved={np.count_nonzero(data != original)}')
