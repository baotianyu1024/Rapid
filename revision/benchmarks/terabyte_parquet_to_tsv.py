"""Export a bounded Criteo Click Logs Parquet slice to the original TSV layout."""
import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--rows', type=int, default=1048576)
a = p.parse_args()
if a.output.exists():
    raise SystemExit('Output already exists')
if shutil.disk_usage(a.output.parent).free < a.rows * 400 + 5 * 1024**3:
    raise SystemExit('Insufficient disk space including reserve')
files = json.loads(a.manifest.read_text())
count = 0
with a.output.open('x', buffering=1024 * 1024) as out:
    for entry in files:
        path = a.manifest.parent / Path(entry['path']).name
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
            cols = [c.to_pylist() for c in batch.columns]
            n = min(batch.num_rows, a.rows - count)
            for row in zip(*(c[:n] for c in cols)):
                out.write('\t'.join('' if v is None else str(v) for v in row) + '\n')
            count += n
            if count == a.rows:
                break
        if count == a.rows:
            break
if count != a.rows:
    raise SystemExit(f'Only {count} rows were available')
print(f'Wrote {count} rows to {a.output}')
