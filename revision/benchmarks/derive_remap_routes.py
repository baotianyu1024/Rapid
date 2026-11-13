"""Derive exact Unique sender/receiver matrices for paired future windows."""

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--result', type=Path,
                    default=Path('/tmp/rapid_remap_crossover_v2_20261113/result.json'))
args = parser.parse_args()
result_path = args.result
ROOT = result_path.parent
result = json.loads(result_path.read_text())
lengths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5,
           1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
width = 14 + sum(lengths)
start = result['initial_rows']
rows = result['new_rows']
if result.get('new_source'):
    raw = np.memmap(result['new_source'], dtype='<u4', mode='r').reshape(-1, width)[:rows]
else:
    raw = np.memmap(result['source'], dtype='<u4', mode='r').reshape(-1, width)[start:start + rows]


def route(name):
    final = ROOT / f'{name}_final'
    assignment = np.load(final / 'assignment.npy')[-rows:]
    count = np.zeros((8, 8), np.int64)
    col = 14
    for table, length in enumerate(lengths):
        ids = np.load(final / f'{table}.original_ids.npy')
        owners = np.load(final / f'{table}.owners.npy')
        for receiver in range(8):
            seen = np.unique(raw[assignment == receiver, col:col + length])
            ix = np.searchsorted(ids, seen)
            if np.any(ix >= len(ids)) or not np.array_equal(ids[ix], seen):
                raise ValueError(f'{name} table {table} misses a future ID')
            source = owners[ix]
            count[:, receiver] += np.bincount(source, minlength=8)
        col += length
    np.fill_diagonal(count, 0)
    return count


incremental = route('incremental')
rebuild = route('rebuild')
assert int(incremental.sum()) == result['incremental_forward_remote_vectors']
assert int(rebuild.sum()) == result['full_rebuild_forward_remote_vectors']
movement = np.zeros((8, 8), np.int64)
for table in range(len(lengths)):
    old = ROOT / 'old_final'
    rebuilt = ROOT / 'rebuild_final'
    old_ids = np.load(old / f'{table}.original_ids.npy')
    old_owners = np.load(old / f'{table}.owners.npy')
    rebuilt_ids = np.load(rebuilt / f'{table}.original_ids.npy')
    rebuilt_owners = np.load(rebuilt / f'{table}.owners.npy')
    positions = np.searchsorted(rebuilt_ids, old_ids)
    if np.any(positions >= len(rebuilt_ids)) or not np.array_equal(rebuilt_ids[positions], old_ids):
        raise ValueError(f'rebuild misses an old ID in table {table}')
    np.add.at(movement, (old_owners, rebuilt_owners[positions]), 1)
np.fill_diagonal(movement, 0)
assert int(movement.sum()) == result['full_rebuild_old_rows_moved']
result['incremental_forward_routes'] = incremental.tolist()
result['rebuild_forward_routes'] = rebuild.tolist()
result['old_row_cross_gpu_routing_counts'] = movement.tolist()
result_path.write_text(json.dumps(result, indent=2) + '\n')
print('incremental', incremental.sum(), 'rebuild', rebuild.sum())
