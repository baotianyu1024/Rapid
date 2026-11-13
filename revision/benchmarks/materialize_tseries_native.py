"""Write active-ID compact HugeCTR RawAsync input for one official T shard."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    meta = json.loads((args.data / 'input.json').read_text())
    rows = meta['rows']
    widths = meta['lengths']
    offsets = meta['active_id_offsets']
    dense = meta['dense']
    record_width = 1 + dense + sum(widths)
    needed = rows * record_width * 4
    if shutil.disk_usage(args.out.parent).free < needed + 8 * 1024**3:
        raise OSError(f'insufficient space before writing {needed:,} bytes')
    raw = np.memmap(args.data / 'input.bin', dtype=np.int32, mode='r',
                    shape=(rows, record_width))
    sparse = np.load(args.data / 'sparse.npy', mmap_mode='r')
    target = np.memmap(args.out, dtype=np.int32, mode='w+', shape=(rows, record_width))
    columns = np.cumsum([0] + widths)
    for start in range(0, rows, 512):
        end = min(rows, start + 512)
        target[start:end, :1 + dense] = raw[start:end, :1 + dense]
        for table in range(len(widths)):
            left, right = columns[table:table + 2]
            target[start:end, 1 + dense + left:1 + dense + right] = (
                sparse[start:end, left:right] - offsets[table])
    target.flush()
    sizes = np.diff(offsets).tolist()
    for table in range(len(widths)):
        left, right = columns[table:table + 2]
        values = target[:, 1 + dense + left:1 + dense + right]
        if values.min() < 0 or values.max() >= sizes[table]:
            raise ValueError(f'compact ID outside table {table}')
    (args.out.parent / f'{args.out.stem}_sizes.json').write_text(json.dumps(sizes) + '\n')
    print(f'rows={rows} tables={len(widths)} bytes={needed}')


if __name__ == '__main__':
    main()
