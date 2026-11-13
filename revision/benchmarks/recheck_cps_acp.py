"""Recount CPS table-20 remote unique IDs directly from materialized batches."""
import json
from pathlib import Path

import numpy as np

out = Path('/tmp/rapid_acp_recheck')
out.mkdir(exist_ok=True)
lengths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
width = 14 + sum(lengths)
table_start = 14 + sum(lengths[:20])
data = np.memmap('/dev/shm/rapid_read_fair/kaggle/materialized.bin', dtype='<u4', mode='r').reshape(16, 8, 8192, width)
counts = np.empty((16, 8), dtype=np.int64)
for batch in range(16):
    for rank in range(8):
        ids = data[batch, rank, :, table_start:table_start + 100].reshape(-1)
        counts[batch, rank] = np.unique(ids[ids % 8 != rank]).size
old = np.load('/tmp/rapid_fresh_stage/kaggle_8_v2/optimized/peak_samples/cps_counts.npz')['receive'][:, 20, :]
result = dict(counts=counts.tolist(), mean=float(counts.mean()), minimum=int(counts.min()),
              maximum=int(counts.max()), per_batch_mean=counts.mean(axis=1).tolist(),
              equal_prior_profile=bool(np.array_equal(counts, old)),
              max_abs_prior_difference=int(np.abs(counts - old).max()),
              threshold_remote=8192 * 7)
(out / 'table20_recount.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
