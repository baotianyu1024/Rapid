"""Measure first-occurrence and owner locality under the fixed final CPS plan."""
import json
from pathlib import Path
import numpy as np

lengths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
width = 14 + sum(lengths)
start = 14 + sum(lengths[:20])
data = np.memmap('/dev/shm/rapid_read_fair/kaggle/materialized.bin', dtype='<u4', mode='r').reshape(16, 8, 8192, width)
size = json.loads(Path('/tmp/rapid_kaggle_plan_sizes.json').read_text())[20]
seen = np.zeros(size, dtype=bool)
rows = []
for batch in range(16):
    batch_unique = np.unique(data[batch, :, :, start:start + 100])
    new_mask = ~seen[batch_unique]
    row = dict(batch=batch, global_unique=len(batch_unique), first_seen=int(new_mask.sum()),
               remote_new=0, remote_seen=0, local_new=0, local_seen=0)
    for rank in range(8):
        ids = np.unique(data[batch, rank, :, start:start + 100])
        remote = ids % 8 != rank
        new = ~seen[ids]
        row['remote_new'] += int(np.count_nonzero(remote & new))
        row['remote_seen'] += int(np.count_nonzero(remote & ~new))
        row['local_new'] += int(np.count_nonzero(~remote & new))
        row['local_seen'] += int(np.count_nonzero(~remote & ~new))
    seen[batch_unique] = True
    rows.append(row)
    print(batch, row, flush=True)
Path('/tmp/rapid_kaggle_novelty_audit.json').write_text(json.dumps(rows, indent=2) + '\n')
