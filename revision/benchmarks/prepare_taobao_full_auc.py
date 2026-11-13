"""Prepare a disjoint full Taobao train/eval split outside the home disk."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


SOURCE = Path('/tmp/rapid_fresh_stage/taobao.bin')
OUT = Path('/dev/shm/rapid_taobao_full_auc_20261109')
RECORD_BYTES = 24
GLOBAL_BATCH = 65536
EVAL_BATCHES = 40
RESERVE_BYTES = 8 * 1024**3


def main():
    rows = SOURCE.stat().st_size // RECORD_BYTES
    if rows * RECORD_BYTES != SOURCE.stat().st_size:
        raise ValueError('partial Taobao record')
    full_batches = rows // GLOBAL_BATCH
    train_rows = (full_batches - EVAL_BATCHES) * GLOBAL_BATCH
    eval_rows = EVAL_BATCHES * GLOBAL_BATCH
    if train_rows <= 0:
        raise ValueError('not enough rows for disjoint split')
    OUT.mkdir(exist_ok=True)
    train_path = OUT / 'train.bin'
    eval_path = OUT / 'eval.bin'
    for destination, first_row, split_rows in ((train_path, 0, train_rows),
                                               (eval_path, train_rows, eval_rows)):
        needed = split_rows * RECORD_BYTES
        if destination.exists():
            if destination.stat().st_size != needed:
                raise ValueError(f'incomplete existing split: {destination}')
            continue
        if shutil.disk_usage(OUT).free < needed + RESERVE_BYTES:
            raise OSError(f'insufficient /dev/shm space before writing {destination}')
        with SOURCE.open('rb') as src, destination.open('xb') as dst:
            src.seek(first_row * RECORD_BYTES)
            left = needed
            while left:
                chunk = src.read(min(left, 8 * 1024**2))
                if not chunk:
                    raise EOFError(f'source ended while writing {destination}')
                dst.write(chunk)
                left -= len(chunk)

    data = np.memmap(SOURCE, dtype='<u4', mode='r').reshape(rows, 6)
    train = data[:train_rows]
    val = data[train_rows:train_rows + eval_rows]
    sizes = [int(max(train[:, col].max(), val[:, col].max())) + 1 for col in range(2, 6)]
    (OUT / 'sizes.json').write_text(json.dumps(sizes) + '\n')
    digest = hashlib.sha256()
    with SOURCE.open('rb') as src:
        for chunk in iter(lambda: src.read(8 * 1024**2), b''):
            digest.update(chunk)
    manifest = dict(source=str(SOURCE), source_sha256=digest.hexdigest(),
                    total_rows=rows, train_rows=train_rows, eval_rows=eval_rows,
                    ignored_tail_rows=rows - train_rows - eval_rows,
                    train_source=str(train_path), eval_source=str(eval_path),
                    table_sizes=sizes,
                    train_positive_rate=float(np.mean(train[:, 0])),
                    eval_positive_rate=float(np.mean(val[:, 0])))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
