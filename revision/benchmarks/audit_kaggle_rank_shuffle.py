"""Redistribute fixed rank-assigned samples across batches as a diagnostic."""
import json
from pathlib import Path
import numpy as np

lengths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
width = 14 + sum(lengths)
start = 14 + sum(lengths[:20])
data = np.memmap('/dev/shm/rapid_read_fair/kaggle/materialized.bin', dtype='<u4', mode='r').reshape(16, 8, 8192, width)
results = {}
for seed in (42, 123, 2026):
    counts = np.zeros((16, 8), dtype=np.int64)
    for rank in range(8):
        samples = np.ascontiguousarray(data[:, rank, :, start:start + 100]).reshape(16 * 8192, 100)
        permutation = np.random.default_rng(seed * 8 + rank).permutation(len(samples))
        for batch in range(16):
            ids = samples[permutation[batch * 8192:(batch + 1) * 8192]].reshape(-1)
            counts[batch, rank] = np.unique(ids[ids % 8 != rank]).size
    results[str(seed)] = dict(counts=counts.tolist(), batch_means=counts.mean(axis=1).tolist(),
                              overall_mean=float(counts.mean()),
                              min_batch_mean=float(counts.mean(axis=1).min()),
                              max_batch_mean=float(counts.mean(axis=1).max()),
                              note='diagnostic: shuffled within each preassigned rank; changes global batch membership, fixed owner and rank')
    print(seed, results[str(seed)]['overall_mean'], results[str(seed)]['min_batch_mean'],
          results[str(seed)]['max_batch_mean'], flush=True)
Path('/tmp/rapid_kaggle_rank_shuffle_audit.json').write_text(json.dumps(results, indent=2) + '\n')
