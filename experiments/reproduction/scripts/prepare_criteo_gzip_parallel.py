"""Materialize the original Criteo multi-hot format with parallel chunk parsing.

Uses the original parser, transform, record dtype, and the cached seed-0 lookup
generated from its table sizes. The final permutation is exactly RandomState(42).
The source is known truncated; this script records that fact rather than calling
the resulting binary a complete Criteo day.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'original_criteo_multihot',
    ROOT/'code/update_Rapid/data_preparation/criteo_singlehot_to_multihot.py')
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)
LOOKUPS = None
OFFSETS = ORIGINAL.build_sparse_offsets()


def init_worker(cache):
    global LOOKUPS
    LOOKUPS = [np.load(Path(cache)/f'table_{t}.npy', mmap_mode='r') for t in range(26)]


def convert_chunk(job):
    idx, lines, part = job
    labels, dense, sparse = ORIGINAL.parse_chunk(lines, idx*len(lines))
    dense_min = int(dense.min())
    if dense_min < -1:
        raise ValueError(f'Unexpected dense minimum {dense_min}; global min requires rescan')
    records = np.empty(len(lines), dtype=ORIGINAL.RECORD_DTYPE)
    records['label'] = labels
    records['dense'] = ORIGINAL.transform_dense(dense, -1)
    records['sparse'] = ORIGINAL.expand_sparse_to_multi_hot(sparse, LOOKUPS, OFFSETS)
    records.tofile(part)
    return idx, part, len(lines), dense_min


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--chunk-rows', type=int, default=100000)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    t0 = time.time()
    source_hash = sha256(args.input)
    temp = args.output.with_suffix('.unshuffled.bin')
    parts = args.output.parent/(args.output.stem+'_parts')
    parts.mkdir(exist_ok=False)
    rows = 0
    minimum = 10**9

    def jobs():
        with gzip.open(args.input, 'rt', errors='ignore') as f:
            idx = 0
            while True:
                lines = ORIGINAL.read_line_chunk(f, None, idx*args.chunk_rows,
                                                 args.chunk_rows)
                if not lines:
                    break
                yield idx, lines, parts/f'{idx:05d}.bin'
                idx += 1

    try:
        with mp.get_context('fork').Pool(args.workers, initializer=init_worker,
                                         initargs=(str(args.cache),)) as pool, temp.open('wb') as dest:
            for expected, (idx, part, n, chunk_min) in enumerate(
                    pool.imap(convert_chunk, jobs(), chunksize=1)):
                if idx != expected:
                    raise RuntimeError('Chunk order changed')
                minimum = min(minimum, chunk_min)
                with part.open('rb') as src:
                    shutil.copyfileobj(src, dest, 16 << 20)
                part.unlink()
                rows += n
                print(f'converted {rows:,} rows in {time.time()-t0:.1f}s', flush=True)
        if minimum != -1:
            raise ValueError(f'Expected dense global minimum -1, observed {minimum}')
        if temp.stat().st_size != rows*ORIGINAL.RECORD_SIZE_BYTES:
            raise RuntimeError('Record size mismatch')
        print(f'globally shuffling {rows:,} rows (seed 42)', flush=True)
        permutation = np.random.RandomState(42).permutation(rows)
        source = np.memmap(temp, dtype=ORIGINAL.RECORD_DTYPE, mode='r')
        with args.output.open('wb') as dest:
            for start in range(0, rows, args.chunk_rows):
                source[permutation[start:start+args.chunk_rows]].tofile(dest)
                if start//args.chunk_rows % 20 == 0:
                    print(f'shuffled {min(start+args.chunk_rows,rows):,}/{rows:,}', flush=True)
        del source, permutation
        if args.output.stat().st_size != rows*ORIGINAL.RECORD_SIZE_BYTES:
            raise RuntimeError('Shuffled record size mismatch')
        metadata = dict(input=str(args.input.resolve()), input_sha256=source_hash,
                        input_gzip_integrity='truncated; gzip -t fails',
                        rows=rows, dense_min=minimum,
                        lookup_seed=0, shuffle_seed=42,
                        table_lengths=ORIGINAL.MULTI_HOT_SIZES,
                        record_bytes=ORIGINAL.RECORD_SIZE_BYTES,
                        output=str(args.output.resolve()),
                        elapsed_seconds=time.time()-t0)
        args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps(metadata, indent=2), flush=True)
    finally:
        if temp.exists():
            temp.unlink()
        if parts.exists():
            shutil.rmtree(parts)


if __name__ == '__main__':
    main()
