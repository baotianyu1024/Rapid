"""Build GMP arrays and a balanced upstream partition for a fresh Criteo trace."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def check_space(directory, needed):
    if shutil.disk_usage(directory).free < needed + 8 * 1024**3:
        raise OSError(f'Insufficient free space in {directory} for {needed:,} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--partition-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.case / 'data/input.json').read_text())
    rows = meta['rows']
    raw = np.memmap(args.case / 'data/input.bin', dtype=np.int32, mode='r').reshape(rows, -1)
    sparse = np.load(args.case / 'data/sparse.npy', mmap_mode='r')
    check_space(args.output, sparse.nbytes + rows * 14 * 4 + 256 * 1024**2)
    for name, data in (('sparse', sparse), ('dense', raw[:, 1:14].copy().view(np.float32)),
                       ('labels', raw[:, 0].astype(np.float32).reshape(-1, 1))):
        check_space(args.output, data.nbytes)
        np.save(args.output / f'{name}.npy', data)
    assignment = np.load(args.partition_dir / 'assignment.npy')
    owner = np.load(args.partition_dir / 'owner.npy')
    counts = np.bincount(assignment, minlength=8)
    balanced = assignment.copy()
    target = rows // 8
    for donor in range(8):
        excess = int(counts[donor] - target)
        if excess <= 0:
            continue
        candidates = np.flatnonzero(balanced == donor)[-excess:]
        moved = 0
        for recipient in range(8):
            count = min(excess - moved, max(0, int(target - counts[recipient])))
            if count:
                balanced[candidates[moved:moved+count]] = recipient
                counts[recipient] += count
                counts[donor] -= count
                moved += count
        if moved != excess:
            raise ValueError('Could not balance GMP ranks')
    if not np.all(np.bincount(balanced, minlength=8) == target):
        raise ValueError('Unbalanced GMP partition')
    arrays = {'data_partition': balanced.astype(np.int32), 'embed_partition': owner.astype(np.int32)}
    for rank in range(8):
        arrays[str(rank)] = np.load(args.partition_dir / f'mirrors_{rank}.npy')
    check_space(args.output, sum(value.nbytes for value in arrays.values()))
    np.savez_compressed(args.output / 'partition_balanced.npz', **arrays)
    print(json.dumps(dict(rows=rows, vocab=len(owner), moved=int(np.count_nonzero(assignment != balanced)),
                          counts=np.bincount(balanced, minlength=8).tolist())), flush=True)


if __name__ == '__main__':
    main()
