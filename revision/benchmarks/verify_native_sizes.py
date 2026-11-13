"""Derive HugeCTR table bounds from both training and evaluation records."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


HOTNESS = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5,
           1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
BASE = Path('/dev/shm/rapid_rw_holdout_20261103')
OUTPUT = Path('/tmp/rapid_rw_e2e_20261103')
OLD = {
    'kaggle': Path('/tmp/rapid_fresh_stage/hugectr_controls/native_sizes.json'),
    'terabyte': Path('/tmp/rapid_fresh_stage/hugectr_terabyte_controls/native_sizes.json'),
    'taobao': Path('/tmp/rapid_rw_e2e_20261103/taobao/native_sizes.json'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('kaggle', 'terabyte', 'taobao'))
    args = parser.parse_args()
    hotness = [1] * 4 if args.dataset == 'taobao' else HOTNESS
    dense = 1 if args.dataset == 'taobao' else 13
    width = 1 + dense + sum(hotness)
    maxima = np.zeros(len(hotness), dtype=np.uint64)
    for split in ('train', 'eval'):
        path = BASE / args.dataset / f'native_{split}.bin'
        records = np.memmap(path, dtype='<u4', mode='r').reshape(-1, width)
        for start in range(0, len(records), 8192):
            block = records[start:start + 8192]
            column = 1 + dense
            for table, count in enumerate(hotness):
                maxima[table] = max(maxima[table], int(block[:, column:column + count].max()))
                column += count
    previous = json.loads(OLD[args.dataset].read_text())
    sizes = [max(int(value) + 1, int(bound)) for value, bound in zip(maxima, previous)]
    destination = OUTPUT / args.dataset / 'native_sizes_verified.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < 8 * 1024**3 + 4096:
        raise OSError(f'insufficient space to write {destination}')
    destination.write_text(json.dumps(sizes) + '\n')
    print(args.dataset, sizes, flush=True)


if __name__ == '__main__':
    main()
