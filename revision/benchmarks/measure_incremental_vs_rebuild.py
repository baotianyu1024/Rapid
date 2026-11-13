"""Compare frozen-old-row placement with full rebuilding on identical Criteo batches."""

import argparse
import ctypes
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[2] / 'experiments'
OPT = EXPERIMENTS / 'reproduction/rapid_peak_optimization'
LENGTHS = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5,
           1, 1, 1, 12, 100, 27, 10, 3, 1, 1]


def received(state, data, assignment, local_batch, choice):
    ranks = state.ranks
    batches = len(data) // (ranks * local_batch)
    total = 0
    per_step = []
    for batch in range(batches):
        rows = data[batch * ranks * local_batch:(batch + 1) * ranks * local_batch]
        owners = assignment[batch * ranks * local_batch:(batch + 1) * ranks * local_batch]
        rank_count = np.zeros(ranks, np.int64)
        col = 14
        for table, length in enumerate(LENGTHS):
            for rank in range(ranks):
                ids = np.unique(rows[owners == rank, col:col + length])
                remote = np.count_nonzero(state.owner[table][ids] != rank)
                rank_count[rank] += remote if choice[table] else local_batch * (ranks - 1)
            col += length
        total += int(rank_count.sum())
        per_step.append(rank_count.tolist())
    return total, per_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=Path('/tmp/rapid_fresh_stage/day6_5767168.bin'))
    parser.add_argument('--new-input', type=Path,
                        help='Use a later-date input for the appended window')
    parser.add_argument('--out', type=Path,
                        default=Path('/tmp/rapid_remap_crossover_20261113'))
    parser.add_argument('--local-batch', type=int, default=2048)
    parser.add_argument('--initial-batches', type=int, default=2)
    parser.add_argument('--new-batches', type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists; choose a fresh directory')
    if shutil.disk_usage(args.out.parent).free < 10 * 1024**3:
        raise OSError('insufficient space for planner output and 10 GiB reserve')
    width = 14 + sum(LENGTHS)
    n0 = 8 * args.local_batch * args.initial_batches
    n1 = 8 * args.local_batch * args.new_batches
    raw = np.memmap(args.input, dtype='<u4', mode='r').reshape(-1, width)
    if len(raw) < n0 + (0 if args.new_input else n1):
        parser.error('input has too few rows')
    old = np.ascontiguousarray(raw[:n0])
    if args.new_input:
        later = np.memmap(args.new_input, dtype='<u4', mode='r').reshape(-1, width)
        if len(later) < n1:
            parser.error('new input has too few rows')
        new = np.ascontiguousarray(later[:n1])
    else:
        new = np.ascontiguousarray(raw[n0:n0 + n1])
    combined = np.concatenate((old, new))
    cfg = dict(batch_size=args.local_batch, dense=13, table_lengths=LENGTHS)
    sys.path[:0] = [str(EXPERIMENTS / 'incremental'), str(OPT)]
    from stream_owner_state import OwnerState
    from wide_stream_window import window
    from wide_stream_alternate import alternate
    import group_scheduler as group
    lib = ctypes.CDLL(str(OPT / 'sample_group_peak_phase.so'))
    signature = group.fn.argtypes
    group.fn = lib.sample_group_peak
    group.fn.argtypes = signature
    group.fn.restype = ctypes.c_int
    args.out.mkdir()

    def plan(state, data, name):
        start = time.perf_counter()
        base = args.out / f'{name}_initial'
        final = args.out / f'{name}_final'
        window(state, data, cfg, base, rounds=4, balance=4, schedule='grouped')
        alternate(state, base, data, final, rounds=4, balance=4,
                  reallocate=True, scheduler='group')
        return time.perf_counter() - start, np.load(final / 'assignment.npy')

    incremental = OwnerState(8, len(LENGTHS))
    initial_seconds, _ = plan(incremental, old, 'old')
    old_ids = [np.flatnonzero(owner >= 0) for owner in incremental.owner]
    old_owner = [owner[ids].copy() for owner, ids in zip(incremental.owner, old_ids)]
    old_physical = [physical[ids].copy() for physical, ids in zip(incremental.physical, old_ids)]
    update_seconds, update_assignment = plan(incremental, new, 'incremental')
    assert all(np.array_equal(incremental.owner[t][ids], old_owner[t]) and
               np.array_equal(incremental.physical[t][ids], old_physical[t])
               for t, ids in enumerate(old_ids))
    full = OwnerState(8, len(LENGTHS))
    rebuild_seconds, full_assignment = plan(full, combined, 'rebuild')
    moved = 0
    repacked = 0
    local_repacked = 0
    for t, ids in enumerate(old_ids):
        changed_owner = full.owner[t][ids] != old_owner[t]
        changed_physical = full.physical[t][ids] != old_physical[t]
        moved += int(np.count_nonzero(changed_owner))
        repacked += int(np.count_nonzero(changed_physical))
        local_repacked += int(np.count_nonzero(changed_physical & ~changed_owner))
    # Use the same all-Unique protocol so the placement comparison is paired.
    choice = [True] * len(LENGTHS)
    update_count, update_steps = received(incremental, new, update_assignment,
                                           args.local_batch, choice)
    full_count, full_steps = received(full, new, full_assignment[-n1:],
                                     args.local_batch, choice)
    delta = update_count - full_count
    # FP32 weights and the AdaGrad accumulator travel together if ownership changes.
    migration_bytes = moved * 32 * 4 * 2
    result = dict(source=str(args.input), new_source=str(args.new_input) if args.new_input else None,
                  ranks=8, dim=32,
                  local_batch=args.local_batch, initial_batches=args.initial_batches,
                  new_batches=args.new_batches, initial_rows=n0, new_rows=n1,
                  initial_plan_seconds=initial_seconds,
                  incremental_update_seconds=update_seconds,
                  full_rebuild_seconds=rebuild_seconds,
                  incremental_old_rows_moved=0, full_rebuild_old_rows_moved=moved,
                  full_rebuild_old_rows_repacked=repacked,
                  full_rebuild_same_owner_rows_repacked=local_repacked,
                  full_rebuild_migration_bytes=migration_bytes,
                  full_rebuild_local_repack_bytes=local_repacked * 32 * 4 * 2,
                  fixed_unique_tables=sum(choice),
                  incremental_forward_remote_vectors=update_count,
                  full_rebuild_forward_remote_vectors=full_count,
                  forward_vectors_saved_per_future_step=delta / args.new_batches,
                  incremental_rank_counts=update_steps,
                  rebuild_rank_counts=full_steps,
                  crossover_future_steps_if_same_vector_bandwidth=(
                      math.ceil(migration_bytes / (2 * delta * 32 * 4))
                      if delta > 0 else None),
                  crossover_scope='transfer-byte equivalent only; excludes plan CPU time and metadata')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
