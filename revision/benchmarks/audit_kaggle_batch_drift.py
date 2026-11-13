"""Compare table-20 access patterns before and after the existing CPS plan."""
import json
from pathlib import Path
import numpy as np

root = Path('/dev/shm/rapid_read_fair/kaggle')
out = Path('/tmp/rapid_kaggle_drift_audit.json')
lengths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
width = 14 + sum(lengths)
start = 14 + sum(lengths[:20])
shape = (16, 8, 8192, width)
native = np.memmap(root / 'native.bin', dtype='<u4', mode='r').reshape(shape)
cps = np.memmap(root / 'materialized.bin', dtype='<u4', mode='r').reshape(shape)
rows = []
for batch in range(16):
    record = {'batch': batch}
    for name, source in [('native', native), ('cps', cps)]:
        values = source[batch, :, :, start:start + 100]
        record[name + '_global_unique'] = int(np.unique(values).size)
        local, remote, total, occurrences = [], [], [], []
        for rank in range(8):
            x = values[rank].reshape(-1)
            unique = np.unique(x)
            local.append(int(np.count_nonzero(unique % 8 == rank)))
            remote.append(int(np.count_nonzero(unique % 8 != rank)))
            total.append(int(unique.size))
            occurrences.append(float(np.mean(x % 8 == rank)))
        record[name + '_unique_per_rank'] = total
        record[name + '_local_unique_per_rank'] = local
        record[name + '_remote_unique_per_rank'] = remote
        record[name + '_local_occurrence_fraction'] = occurrences
    record['global_unique_equal'] = record['native_global_unique'] == record['cps_global_unique']
    rows.append(record)
    print(batch, record['native_global_unique'], round(np.mean(record['native_remote_unique_per_rank']), 1),
          round(np.mean(record['cps_remote_unique_per_rank']), 1), record['global_unique_equal'], flush=True)
out.write_text(json.dumps(rows, indent=2) + '\n')
