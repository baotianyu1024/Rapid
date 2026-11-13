"""Map an independent Criteo validation day into training ID spaces."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--eval-input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output directory must not exist')
    meta = json.loads((args.case / 'data/input.json').read_text())
    lengths = meta['lengths']
    width = 14 + sum(lengths)
    source = np.memmap(args.eval_input, dtype='<u4', mode='r').reshape(-1, width)
    rows = len(source)
    if rows % (meta['ranks'] * meta['local_batch']):
        raise ValueError('evaluation rows must divide into global batches')
    needed = rows * width * 4 + rows * sum(lengths) * 4 + 4 * 1024**3
    free = shutil.disk_usage(args.out.parent).free
    if free < needed:
        raise OSError(f'insufficient space: free={free}, need={needed}')
    args.out.mkdir()
    torchrec_dir = args.out / 'torchrec_eval'
    torchrec_dir.mkdir()
    (torchrec_dir / 'input.bin').symlink_to(args.eval_input.resolve())
    eval_meta = dict(meta, rows=rows, global_batches=rows // (meta['ranks'] * meta['local_batch']),
                     scope='independent validation date; unseen logical IDs mapped to table index zero')
    (torchrec_dir / 'input.json').write_text(json.dumps(eval_meta, indent=2) + '\n')
    sparse = np.lib.format.open_memmap(torchrec_dir / 'sparse.npy', mode='w+',
                                       dtype=np.int32, shape=(rows, sum(lengths)))
    native = np.memmap(args.out / 'native_eval.bin', dtype='<u4', mode='w+', shape=(rows, width))
    native[:, :14] = source[:, :14]
    offsets = meta['active_id_offsets']
    col = 0
    oov = []
    for table, length in enumerate(lengths):
        known = np.load(args.case / 'data' / f'ids_{table}.npy', mmap_mode='r')
        missing = 0
        for first in range(0, rows, 8192):
            values = source[first:first + 8192, 14 + col:14 + col + length]
            pos = np.searchsorted(known, values)
            safe = np.minimum(pos, len(known) - 1)
            seen = (pos < len(known)) & (known[safe] == values)
            missing += int((~seen).sum())
            mapped = np.where(seen, pos, 0).astype(np.int32)
            sparse[first:first + len(values), col:col + length] = mapped + offsets[table]
            native[first:first + len(values), 14 + col:14 + col + length] = mapped
        oov.append(missing)
        col += length
    sparse.flush()
    native.flush()
    (args.out / 'native_sizes.json').write_text(json.dumps(
        [end - start for start, end in zip(offsets, offsets[1:])]) + '\n')
    (args.out / 'audit.json').write_text(json.dumps(dict(rows=rows, oov_by_table=oov,
        total_oov=sum(oov), sparse_slots=rows * sum(lengths),
        note='Unseen validation IDs use first known training ID in each table'), indent=2) + '\n')
    print('mapped validation', rows, 'rows; OOV', sum(oov), 'of', rows * sum(lengths), flush=True)


if __name__ == '__main__':
    main()
