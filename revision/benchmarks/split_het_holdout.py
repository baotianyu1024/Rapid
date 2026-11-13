"""Hold out the final global batch from in-memory HET-GMP arrays."""
import argparse
import os
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--global-batch', type=int, default=65536)
a = p.parse_args()

train = a.out / 'train'
valid = a.out / 'valid'
train.mkdir(parents=True, exist_ok=True)
valid.mkdir(parents=True, exist_ok=True)
for name in ('dense', 'sparse', 'labels'):
    array = np.load(a.source / f'{name}.npy', mmap_mode='r')
    if len(array) < 2 * a.global_batch or len(array) % a.global_batch:
        raise ValueError('expected at least two complete global batches')
    for folder, view in ((train, array[:-a.global_batch]),
                         (valid, array[-a.global_batch:])):
        target = folder / f'{name}.npy'
        if target.exists():
            raise FileExistsError(target)
        stat = os.statvfs(folder)
        free = stat.f_bavail * stat.f_frsize
        if free < view.nbytes + 2**30:
            raise OSError(f'insufficient free space for {target}: {free} bytes')
        np.save(target, view)
print(f'train_rows={len(array) - a.global_batch} valid_rows={a.global_batch}')
