"""Prepare MovieLens-25M implicit labels with a temporal holdout."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path('/tmp/rapid_movielens25m_parallel_20261103/ml-25m/ratings.csv')
OUT = Path('/dev/shm/rapid_movielens25m_temporal_20261103')
BATCH = 65536
HEADROOM = 8 * 1024**3


def require_space(needed):
    free = shutil.disk_usage(OUT).free
    if free < needed + HEADROOM:
        raise OSError(f'insufficient space at {OUT}: free={free}, need={needed + HEADROOM}')


def write_split(name, indices, user, movie, rating, offsets, cutoff):
    destination = OUT / name
    destination.mkdir(parents=True, exist_ok=True)
    rows = len(indices)
    raw_path = destination / 'input.bin'
    sparse_path = destination / 'sparse.npy'
    require_space(rows * 16)
    raw = np.memmap(raw_path, dtype='<u4', mode='w+', shape=(rows, 4))
    require_space(rows * 16)
    sparse = np.lib.format.open_memmap(sparse_path, mode='w+', dtype='<i8', shape=(rows, 2))
    for start in range(0, rows, 1_000_000):
        end = min(rows, start + 1_000_000)
        source = indices[start:end]
        user_ids = user[source]
        movie_ids = movie[source]
        raw[start:end, 0] = (rating[source] >= 4).astype('<u4')
        raw[start:end, 1] = 0
        raw[start:end, 2] = user_ids
        raw[start:end, 3] = movie_ids
        sparse[start:end, 0] = user_ids
        sparse[start:end, 1] = movie_ids.astype(np.int64) + offsets[1]
    raw.flush()
    sparse.flush()
    positives = int((rating[indices] >= 4).sum())
    metadata = dict(dataset='movielens25m', rows=rows, global_batches=rows // BATCH,
                    ranks=8, local_batch=8192, lengths=[1, 1], dense=1, dim=64,
                    record_bytes=16, active_id_offsets=offsets,
                    temporal_cutoff=int(cutoff) if cutoff is not None else None,
                    label='rating >= 4',
                    positives=positives, source=str(SOURCE))
    (destination / 'input.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(name, rows, positives, flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-user', action='store_true', help='Hold out each user\'s latest 20%')
    args = parser.parse_args()
    if args.per_user:
        OUT = Path('/dev/shm/rapid_movielens25m_user_temporal_20261103')
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE, usecols=['userId', 'movieId', 'rating', 'timestamp'],
                        dtype={'userId': 'uint32', 'movieId': 'uint32',
                               'rating': 'float32', 'timestamp': 'int64'})
    user = frame['userId'].to_numpy()
    movie = frame['movieId'].to_numpy()
    rating = frame['rating'].to_numpy()
    timestamp = frame['timestamp'].to_numpy()
    if args.per_user:
        order = np.lexsort((timestamp, user))
        starts = np.r_[0, np.flatnonzero(user[order][1:] != user[order][:-1]) + 1]
        counts = np.diff(np.r_[starts, len(order)])
        within = np.arange(len(order)) - np.repeat(starts, counts)
        train_mask = within < np.repeat((counts * 4) // 5, counts)
        train_indices = order[train_mask]
        eval_indices = order[~train_mask]
        np.random.default_rng(42).shuffle(train_indices)
        np.random.default_rng(43).shuffle(eval_indices)
        train_rows = len(train_indices) // BATCH * BATCH
        eval_rows = len(eval_indices) // BATCH * BATCH
        cutoff = None
    else:
        order = np.argsort(timestamp, kind='stable')
        train_rows = (int(len(order) * 0.8) // BATCH) * BATCH
        while train_rows >= BATCH and timestamp[order[train_rows - 1]] == timestamp[order[train_rows]]:
            train_rows -= BATCH
        eval_rows = ((len(order) - train_rows) // BATCH) * BATCH
        cutoff = timestamp[order[train_rows - 1]]
        train_indices = order[:train_rows].copy()
        np.random.default_rng(42).shuffle(train_indices)
        eval_indices = order[train_rows:train_rows + eval_rows]
    if train_rows < BATCH or eval_rows < BATCH:
        raise ValueError('not enough complete batches for temporal split')
    user_size = int(user.max()) + 1
    movie_size = int(movie.max()) + 1
    offsets = [0, user_size, user_size + movie_size]
    write_split('train', train_indices[:train_rows], user, movie, rating, offsets, cutoff)
    write_split('eval', eval_indices[:eval_rows], user, movie, rating, offsets, cutoff)
    (OUT / 'table_sizes.json').write_text(json.dumps([user_size, movie_size]) + '\n')
    manifest = dict(source=str(SOURCE), source_rows=len(order), train_rows=train_rows,
                    eval_rows=eval_rows,
                    dropped_train_rows=len(train_indices)-train_rows,
                    dropped_eval_rows=len(eval_indices)-eval_rows,
                    split='per_user_temporal' if args.per_user else 'global_temporal',
                    cutoff_timestamp=int(cutoff) if cutoff is not None else None,
                    shuffle_seed=42, offsets=offsets)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
