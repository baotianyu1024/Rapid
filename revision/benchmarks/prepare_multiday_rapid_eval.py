"""Map a held-out Criteo date through the final RAPID physical-ID plan."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--eval-input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    meta = json.loads((args.case / 'data/input.json').read_text())
    lengths = meta['lengths']
    width = 14 + sum(lengths)
    source = np.memmap(args.eval_input, dtype='<u4', mode='r').reshape(-1, width)
    required = source.nbytes + 4 * 1024**3
    free = shutil.disk_usage(args.out.parent).free
    if free < required:
        raise OSError(f'insufficient output space: free={free}, need={required}')
    output = np.memmap(args.out, dtype='<u4', mode='w+', shape=source.shape)
    output[:, :14] = source[:, :14]
    misses = []
    mapping = (args.plan / 'embedding_mapping.tsv').open()
    current = next(mapping, None)
    col = 0
    for table, length in enumerate(lengths):
        values = source[:, 14 + col:14 + col + length]
        known = np.load(args.case / 'data' / f'ids_{table}.npy', mmap_mode='r')
        capacity = max(int(values.max()), int(known[-1])) + 1
        lut = np.full(capacity, np.iinfo(np.uint32).max, dtype=np.uint32)
        while current is not None:
            parts = current.split()
            if len(parts) != 4:
                raise ValueError('invalid RAPID mapping record')
            owner_table, key, _, physical = map(int, parts)
            if owner_table != table:
                if owner_table < table:
                    raise ValueError('mapping table order is invalid')
                break
            if key >= capacity:
                raise ValueError('mapping key exceeds official vocabulary')
            lut[key] = physical
            current = next(mapping, None)
        fallback = lut[int(known[0])]
        if fallback == np.iinfo(np.uint32).max:
            raise ValueError(f'first known ID is unmapped in table {table}')
        absent = 0
        for first in range(0, len(source), 8192):
            block = values[first:first + 8192]
            physical = lut[block]
            unseen = physical == np.iinfo(np.uint32).max
            absent += int(unseen.sum())
            output[first:first + len(block), 14 + col:14 + col + length] = np.where(
                unseen, fallback, physical)
        misses.append(absent)
        print('table', table, 'oov', absent, flush=True)
        col += length
        del lut
    if current is not None:
        raise ValueError('mapping contains extra tables')
    mapping.close()
    output.flush()
    (args.out.with_suffix('.json')).write_text(json.dumps(dict(rows=len(source),
        oov_by_table=misses, total_oov=sum(misses),
        note='Unseen validation IDs use the physical row of the first known training ID'), indent=2) + '\n')


if __name__ == '__main__':
    main()
