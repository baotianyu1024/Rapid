"""Partition the exact scaled T-series input with HET-GMP's upstream planner."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/tmp/rapid_tseries_scaled_20261112')
PARTITIONER = ROOT / 'code/Rapid/experiments/reproduction/het_gmp_rapid/scripts/partition_het.py'
CASES = {'t180': 128, 't200': 512, 't510': 512}


def check_space(path, needed):
    if shutil.disk_usage(path).free < needed + 8 * 1024**3:
        raise OSError(f'insufficient disk space before writing {needed:,} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=CASES)
    parser.add_argument('--data-root', type=Path, default=BASE)
    args = parser.parse_args()
    case = args.data_root / args.dataset
    source = case / 'torchrec'
    info = json.loads((source / 'input.json').read_text())
    rows = info['rows']
    batch = info['local_batch'] if 'local_batch' in info else CASES[args.dataset]
    if rows % (8 * batch):
        raise ValueError('training rows do not divide into eight equal-rank batches')
    stage = case / 'het'
    partition = stage / 'partition'
    if stage.exists():
        parser.error(f'output already exists: {stage}')
    estimated = (source / 'sparse.npy').stat().st_size + rows * 14 * 4 + 2 * 1024**3
    check_space(case, estimated)
    stage.mkdir()
    (stage / 'sparse.npy').symlink_to(source / 'sparse.npy')
    raw = np.memmap(source / 'input.bin', dtype='<i4', mode='r').reshape(rows, -1)
    np.save(stage / 'dense.npy', raw[:, 1:14].copy().view(np.float32))
    np.save(stage / 'labels.npy', raw[:, 0].astype(np.float32).reshape(-1, 1))
    partition.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS='16', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', str(PARTITIONER), '--data', str(source),
               '--out', str(partition), '--ranks', '8', '--batch-size', str(batch)]
    with (stage / 'partition.log').open('w') as stream:
        subprocess.run(command, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    assignment = np.load(partition / 'assignment.npy')
    owners = np.load(partition / 'owner.npy')
    full_vocab = sum(json.loads((case / 'sizes.json').read_text()))
    if len(owners) > full_vocab:
        raise ValueError('HET-GMP partitioner produced IDs beyond the declared vocabulary')
    if len(owners) < full_vocab:
        missing = np.arange(len(owners), full_vocab, dtype=np.int64) % 8
        owners = np.concatenate((owners, missing.astype(np.int32)))
    target = rows // 8
    balanced = assignment.copy()
    counts = np.bincount(balanced, minlength=8)
    for donor in range(8):
        excess = int(counts[donor] - target)
        if excess <= 0:
            continue
        candidates = np.flatnonzero(balanced == donor)[-excess:]
        cursor = 0
        for recipient in range(8):
            take = min(excess - cursor, max(0, int(target - counts[recipient])))
            if take:
                balanced[candidates[cursor:cursor + take]] = recipient
                counts[recipient] += take
                counts[donor] -= take
                cursor += take
        if cursor != excess:
            raise ValueError('cannot balance HET-GMP sample partition')
    if not np.array_equal(np.bincount(balanced, minlength=8), np.full(8, target)):
        raise ValueError('unbalanced HET-GMP sample partition')
    arrays = {'data_partition': balanced.astype(np.int32),
              'embed_partition': owners.astype(np.int32)}
    for rank in range(8):
        arrays[str(rank)] = np.load(partition / f'mirrors_{rank}.npy')
    check_space(stage, sum(array.nbytes for array in arrays.values()))
    np.savez_compressed(stage / 'partition_balanced.npz', **arrays)
    print(json.dumps(dict(dataset=args.dataset, rows=rows, tables=len(info['lengths']),
                          full_vocab=len(owners), sample_moves=int(np.count_nonzero(
                              balanced != assignment)))), flush=True)


if __name__ == '__main__':
    main()
