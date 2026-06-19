"""
Fast hot ID extraction using numpy.
Reads multi-hot binary dataset and finds global top-K hottest (table_id, feature_id) pairs.
Outputs per-table files: {table_id}.txt with "feature_id : count" lines.
"""
import numpy as np
import os
import argparse
from collections import Counter
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Fast hot ID extraction using numpy")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--topk', type=int, default=-1, help='Global top-K. -1 = all')
    parser.add_argument('--dense_count', type=int, default=13)
    parser.add_argument('--ignore_tables', type=int, nargs='*', default=[])
    default_lengths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
    parser.add_argument('--table_lengths', type=int, nargs='+', default=default_lengths)
    parser.add_argument('--chunk_rows', type=int, default=500000, help='Rows per chunk')
    parser.add_argument('--max_rows', type=int, default=None, help='Max rows to scan (default: all)')
    return parser.parse_args()


def main():
    args = parse_args()
    table_lengths = args.table_lengths
    num_tables = len(table_lengths)
    total_sparse = sum(table_lengths)
    dense_count = args.dense_count
    ignore_set = set(args.ignore_tables)

    # Build dtype
    dt = np.dtype([
        ('label', 'i4'),
        ('dense', 'f4', (dense_count,)),
        ('sparse', 'i4', (total_sparse,))
    ])
    record_size = dt.itemsize

    file_size = os.path.getsize(args.input)
    total_rows = file_size // record_size
    if args.max_rows is not None:
        total_rows = min(total_rows, args.max_rows)
    print(f"Input: {args.input} ({file_size / 1024**3:.2f} GB, scanning {total_rows} rows)")
    print(f"Tables: {num_tables}, total sparse cols: {total_sparse}")
    print(f"TopK: {'all' if args.topk <= 0 else args.topk}")
    print(f"Ignore tables: {list(ignore_set) if ignore_set else 'none'}")

    # Pre-compute table offsets
    offsets = []
    off = 0
    for tl in table_lengths:
        offsets.append((off, off + tl))
        off += tl

    # Per-table counters
    table_counters = [Counter() for _ in range(num_tables)]

    t0 = time.time()
    processed = 0

    with open(args.input, 'rb') as f:
        while processed < total_rows:
            chunk = min(args.chunk_rows, total_rows - processed)
            data = np.fromfile(f, dtype=dt, count=chunk)
            if len(data) == 0:
                break

            sparse = data['sparse']  # (chunk, total_sparse)

            for t in range(num_tables):
                if t in ignore_set:
                    continue
                start, end = offsets[t]
                cols = sparse[:, start:end].ravel()
                # Count using numpy unique
                vals, counts = np.unique(cols, return_counts=True)
                tc = table_counters[t]
                for v, c in zip(vals, counts):
                    tc[int(v)] += int(c)

            processed += len(data)
            elapsed = time.time() - t0
            speed = processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {processed}/{total_rows} rows ({speed:.0f} rows/s)", end='\r')

    elapsed = time.time() - t0
    print(f"\nCounting done in {elapsed:.1f}s")

    # Merge all counters into global list for topk
    print("Sorting globally...")
    all_items = []
    for t in range(num_tables):
        if t in ignore_set:
            continue
        for fid, cnt in table_counters[t].items():
            all_items.append((t, fid, cnt))

    all_items.sort(key=lambda x: x[2], reverse=True)
    print(f"Total unique (table, feature) pairs: {len(all_items)}")

    # Apply topk
    if args.topk > 0:
        all_items = all_items[:args.topk]
        print(f"Keeping top {args.topk}")

    # Write per-table files
    os.makedirs(args.output, exist_ok=True)
    file_handles = {}
    for t, fid, cnt in all_items:
        if t not in file_handles:
            file_handles[t] = open(os.path.join(args.output, f"{t}.txt"), 'w')
        file_handles[t].write(f"{fid} : {cnt}\n")

    for fh in file_handles.values():
        fh.close()

    # Summary
    print(f"\nOutput: {args.output}")
    for t in sorted(file_handles.keys()):
        fpath = os.path.join(args.output, f"{t}.txt")
        lines = sum(1 for _ in open(fpath))
        print(f"  Table {t:2d}: {lines} hot IDs")

    print("Done.")


if __name__ == '__main__':
    main()
