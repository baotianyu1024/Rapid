"""Adapt a scaled T-series raw shard to the TorchRec input layout."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path('/tmp/rapid_tseries_scaled_20261112')
RESERVE = 8 * 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('t180', 't200', 't510'))
    parser.add_argument('--data-root', type=Path, default=ROOT)
    args = parser.parse_args()
    case = args.data_root / args.dataset
    meta = json.loads((case / 'manifest.json').read_text())
    source = case / ('input_small.bin' if args.dataset == 't180' and meta['rows'] > 8192
                     else 'input.bin')
    rows = 8192 if args.dataset == 't180' and meta['rows'] > 8192 else meta['rows']
    hot = meta['lengths']
    width = 14 + sum(hot)
    if source.stat().st_size != rows * width * 4:
        raise ValueError('raw shard has a different number of records')
    out = case / 'torchrec'
    if out.exists():
        parser.error(f'output already exists: {out}')
    sparse_bytes = rows * sum(hot) * 4 + 4096
    if shutil.disk_usage(case).free < sparse_bytes + RESERVE:
        raise OSError('insufficient space for TorchRec sparse input and reserve')
    out.mkdir()
    (out / 'input.bin').symlink_to(source)
    raw = np.memmap(source, dtype='<i4', mode='r').reshape(rows, width)
    sparse = np.lib.format.open_memmap(out / 'sparse.npy', mode='w+',
                                       dtype=np.int32, shape=(rows, sum(hot)))
    offsets = np.cumsum([0] + meta['sizes'], dtype=np.int64)
    if offsets[-1] >= np.iinfo(np.int32).max:
        raise ValueError('global IDs exceed the int32 TorchRec input format')
    start = 0
    for table, length in enumerate(hot):
        field = raw[:, 14 + start:14 + start + length]
        if field.min() < 0 or field.max() >= meta['sizes'][table]:
            raise ValueError(f'IDs exceed vocabulary size of table {table}')
        sparse[:, start:start + length] = field + offsets[table]
        start += length
    sparse.flush()
    info = dict(dataset=meta['dataset'], rows=rows, dense=13, lengths=hot,
                active_id_offsets=offsets.tolist(), record_bytes=width * 4,
                source_input=str(source))
    serialized = json.dumps(info, indent=2) + '\n'
    if shutil.disk_usage(out).free < len(serialized.encode()) + RESERVE:
        raise OSError('insufficient space before writing TorchRec metadata')
    (out / 'input.json').write_text(serialized)
    print(f'{args.dataset}: {rows} records, {len(hot)} tables, {sparse_bytes} sparse bytes')


if __name__ == '__main__':
    main()
