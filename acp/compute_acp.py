"""
Compute per-table ACP (Adaptive Data Access Planning) strategy selection.

For each table, compares Unique vs Reduction communication cost:
  - Unique cost  = |U_i| = avg remote unique IDs per GPU per batch
  - Reduction cost = B * G (batch_size * num_gpus)

Outputs: per-table strategy, per-table cost, total volume under 4 configs:
  All-Reduction, All-Unique, ACP, ACP+CPS

Usage:
  # Hash partition (original dataset):
  python3 scripts/compute_acp.py --input data.bin --num_gpus 8 32 128 \
      --batch_size 8192 --table_lengths 3 2 1 ... --dense_count 13 --num_batches 50

  # With CPS data:
  python3 scripts/compute_acp.py --input data.bin --num_gpus 8 \
      --batch_size 8192 --table_lengths 3 2 1 ... --dense_count 13 \
      --cps_dataset_dir data/hugectr_train_8gpu/dataset_output --num_batches 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def build_col_to_table(table_lengths):
    col_to_table = []
    for t, tl in enumerate(table_lengths):
        col_to_table.extend([t] * tl)
    return np.array(col_to_table, dtype=np.int32)


def build_table_slices(table_lengths):
    slices = []
    off = 0
    for tl in table_lengths:
        slices.append((off, off + tl))
        off += tl
    return slices


def compute_unique_per_table_hash(input_path, num_gpus, batch_size, num_batches,
                                   table_lengths, dense_count):
    """Compute per-table avg remote unique IDs under hash partition."""
    total_sparse = sum(table_lengths)
    num_tables = len(table_lengths)
    record_ints = 1 + dense_count + total_sparse
    global_batch = num_gpus * batch_size
    slices = build_table_slices(table_lengths)

    dt = np.dtype([
        ('label', 'i4'),
        ('dense', 'f4' if dense_count > 0 else 'i4', (max(dense_count, 1),)),
        ('sparse', 'i4', (total_sparse,))
    ])

    # Per-table: sum of remote unique IDs across all batches and GPUs
    sum_remote = np.zeros(num_tables, dtype=np.float64)
    sum_unique = np.zeros(num_tables, dtype=np.float64)
    actual_batches = 0

    with open(input_path, 'rb') as f:
        for b in range(num_batches):
            data = np.fromfile(f, dtype=dt, count=global_batch)
            if len(data) < global_batch:
                break
            sparse = data['sparse']
            actual_batches += 1

            for t, (start, end) in enumerate(slices):
                table_data = sparse[:, start:end]

                batch_remote = 0
                batch_unique = 0
                for g in range(num_gpus):
                    gs = g * batch_size
                    ge = gs + batch_size
                    gpu_data = table_data[gs:ge].ravel()
                    unique_ids = np.unique(gpu_data)
                    n_unique = len(unique_ids)
                    remote_mask = (unique_ids % num_gpus) != g
                    n_remote = int(remote_mask.sum())
                    batch_remote += n_remote
                    batch_unique += n_unique

                sum_remote[t] += batch_remote / num_gpus
                sum_unique[t] += batch_unique / num_gpus

            if (b + 1) % 10 == 0:
                print(f"  Batch {b+1}/{num_batches}")

    if actual_batches == 0:
        return np.zeros(num_tables), np.zeros(num_tables)

    avg_remote = sum_remote / actual_batches
    avg_unique = sum_unique / actual_batches
    return avg_remote, avg_unique


def compute_unique_per_table_cps(dataset_dir, num_gpus, batch_size, num_batches,
                                  table_lengths, dense_count):
    """Compute per-table avg remote unique IDs from CPS-transformed dataset."""
    total_sparse = sum(table_lengths)
    num_tables = len(table_lengths)
    slices = build_table_slices(table_lengths)
    col_to_table = build_col_to_table(table_lengths)

    dt = np.dtype([
        ('label', 'i4'),
        ('dense', 'f4' if dense_count > 0 else 'i4', (max(dense_count, 1),)),
        ('sparse', 'i4', (total_sparse,))
    ])

    samples_per_dev = batch_size * num_batches
    record_size = dt.itemsize

    sum_remote = np.zeros(num_tables, dtype=np.float64)
    sum_unique = np.zeros(num_tables, dtype=np.float64)
    max_remote_per_table = np.zeros(num_tables, dtype=np.float64)
    actual_batches = 0

    # Read all device files
    dev_data = []
    for d in range(num_gpus):
        fpath = os.path.join(dataset_dir, f"dataset_{d}.bin")
        if not os.path.exists(fpath):
            print(f"  Warning: {fpath} not found, skipping")
            continue
        fsize = os.path.getsize(fpath)
        n_samples = fsize // record_size
        actual_dev_batches = n_samples // batch_size
        dev_data.append((fpath, min(actual_dev_batches, num_batches)))

    if not dev_data:
        return np.zeros(num_tables), np.zeros(num_tables), np.zeros(num_tables)

    actual_batches = min(d[1] for d in dev_data)

    for b in range(actual_batches):
        batch_remote = np.zeros(num_tables)
        batch_unique = np.zeros(num_tables)

        for d, (fpath, _) in enumerate(dev_data):
            data = np.fromfile(fpath, dtype=dt, count=batch_size,
                              offset=b * batch_size * record_size)
            if len(data) < batch_size:
                break
            sparse = data['sparse']

            for t, (start, end) in enumerate(slices):
                table_data = sparse[:, start:end].ravel().astype(np.uint32)
                unique_ids = np.unique(table_data)
                n_unique = len(unique_ids)
                # In CPS, new_id % num_devices == device_id means local
                remote_mask = (unique_ids % num_gpus) != d
                n_remote = int(remote_mask.sum())
                batch_remote[t] += n_remote
                batch_unique[t] += n_unique

                if n_remote > max_remote_per_table[t]:
                    max_remote_per_table[t] = n_remote

        sum_remote += batch_remote / num_gpus
        sum_unique += batch_unique / num_gpus

        if (b + 1) % 10 == 0:
            print(f"  Batch {b+1}/{actual_batches}")

    avg_remote = sum_remote / actual_batches
    avg_unique = sum_unique / actual_batches
    return avg_remote, avg_unique, max_remote_per_table


def compute_acp(avg_remote_unique, table_lengths, batch_size, num_gpus):
    """Compute ACP strategy selection and costs."""
    num_tables = len(table_lengths)
    results = []

    for t in range(num_tables):
        L_i = table_lengths[t]
        c_unique = avg_remote_unique[t]
        c_reduction = batch_size * num_gpus
        strategy = "Unique" if c_unique < c_reduction else "Reduction"
        cost = min(c_unique, c_reduction)
        results.append({
            'table': t,
            'pooling_factor': L_i,
            'c_unique': float(c_unique),
            'c_reduction': float(c_reduction),
            'strategy': strategy,
            'cost': float(cost),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="ACP: per-table strategy selection")
    parser.add_argument("--input", required=True, help="Binary dataset path")
    parser.add_argument("--num_gpus", type=int, nargs='+', default=[8])
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--table_lengths", type=int, nargs='+', required=True)
    parser.add_argument("--dense_count", type=int, default=13)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--cps_dataset_dirs", type=str, nargs='*', default=[],
                        help="CPS dataset dirs, one per num_gpus entry")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dense_float", action='store_true',
                        help="Dense features are float32 (default: auto)")
    args = parser.parse_args()

    table_lengths = args.table_lengths
    num_tables = len(table_lengths)
    B = args.batch_size

    all_results = {}

    for gi, G in enumerate(args.num_gpus):
        print(f"\n{'='*60}")
        print(f"Computing ACP for G={G} GPUs")
        print(f"{'='*60}")

        # Hash partition
        file_size = os.path.getsize(args.input)
        total_sparse = sum(table_lengths)
        record_size = 4 + args.dense_count * 4 + total_sparse * 4
        total_samples = file_size // record_size
        global_batch = G * B
        max_batches = total_samples // global_batch
        nb = min(args.num_batches, max_batches)
        print(f"  File: {total_samples} samples, {max_batches} full batches, using {nb}")

        print(f"\n  [Hash partition] Computing per-table unique...")
        t0 = time.time()
        avg_remote_hash, avg_unique_hash = compute_unique_per_table_hash(
            args.input, G, B, nb, table_lengths, args.dense_count)
        print(f"  Done in {time.time()-t0:.1f}s")

        # ACP decision
        acp_hash = compute_acp(avg_remote_hash, table_lengths, B, G)

        # Totals
        total_reduction = sum(B * G for _ in range(num_tables))
        total_unique = sum(avg_remote_hash)
        total_acp = sum(r['cost'] for r in acp_hash)

        n_reduction = sum(1 for r in acp_hash if r['strategy'] == 'Reduction')
        n_unique = sum(1 for r in acp_hash if r['strategy'] == 'Unique')

        result = {
            'num_gpus': G,
            'batch_size': B,
            'num_batches': nb,
            'all_reduction': float(total_reduction),
            'all_unique': float(total_unique),
            'acp': float(total_acp),
            'acp_n_reduction': n_reduction,
            'acp_n_unique': n_unique,
            'per_table': acp_hash,
        }

        # CPS
        if gi < len(args.cps_dataset_dirs) and args.cps_dataset_dirs[gi]:
            cps_dir = args.cps_dataset_dirs[gi]
            print(f"\n  [CPS partition] Computing per-table unique from {cps_dir}...")
            t0 = time.time()
            avg_remote_cps, avg_unique_cps, max_remote_cps = compute_unique_per_table_cps(
                cps_dir, G, B, nb, table_lengths, args.dense_count)
            print(f"  Done in {time.time()-t0:.1f}s")

            acp_cps = compute_acp(avg_remote_cps, table_lengths, B, G)
            total_acp_cps = sum(r['cost'] for r in acp_cps)
            result['acp_cps'] = float(total_acp_cps)
            result['acp_cps_per_table'] = acp_cps

        # Print summary
        print(f"\n  === G={G} Summary ===")
        print(f"  All-Reduction:  {total_reduction:>12,.0f} vectors")
        print(f"  All-Unique:     {total_unique:>12,.0f} vectors")
        print(f"  ACP:            {total_acp:>12,.0f} vectors ({n_reduction}R + {n_unique}U)")
        if 'acp_cps' in result:
            print(f"  ACP+CPS:        {result['acp_cps']:>12,.0f} vectors")
        print(f"  ACP/Unique:     {total_acp/total_unique*100:.1f}%")
        if 'acp_cps' in result:
            print(f"  ACP+CPS/Unique: {result['acp_cps']/total_unique*100:.1f}%")

        # Print per-table details for tables where Reduction is chosen
        red_tables = [r for r in acp_hash if r['strategy'] == 'Reduction']
        if red_tables:
            print(f"\n  Tables using Reduction:")
            for r in red_tables:
                print(f"    Table {r['table']} (pool={r['pooling_factor']}): "
                      f"unique={r['c_unique']:.0f} vs reduction={r['c_reduction']:.0f}")

        all_results[str(G)] = result

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return all_results


if __name__ == "__main__":
    main()
