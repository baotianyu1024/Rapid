"""Append complete input batches to a RAPID runtime plan without moving old IDs."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--plan', type=Path, required=True)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--ranks', type=int, required=True)
p.add_argument('--local-batch', type=int, required=True)
p.add_argument('--batches', type=int, required=True)
p.add_argument('--dense', type=int, required=True)
p.add_argument('--table-lengths', type=int, nargs='+', required=True)
p.add_argument('--publish', type=Path, help='Atomically publish this complete plan through a marker file')
a = p.parse_args()
if a.out.exists() or min(a.ranks, a.local_batch, a.batches, *a.table_lengths) < 1 or a.dense < 0:
    p.error('invalid dimensions or output already exists')
global_batch = a.ranks * a.local_batch
indices = [np.fromfile(a.plan / f'sample_indices_{r}.bin', dtype='<u8') for r in range(a.ranks)]
if any(len(x) != len(indices[0]) for x in indices) or len(indices[0]) % a.local_batch:
    p.error('input plan has incomplete or unbalanced batches')
start = len(indices[0]) * a.ranks
width = 1 + a.dense + sum(a.table_lengths)
required = (start + a.batches * global_batch) * width * 4
if a.input.stat().st_size < required:
    p.error('input does not contain all requested batches')
if shutil.disk_usage(a.out.parent).free < (a.plan / 'embedding_mapping.tsv').stat().st_size * 2 + 1024**3:
    p.error('insufficient space for plan and reserve')

mapping = {}
next_local = np.zeros((len(a.table_lengths), a.ranks), dtype=np.uint64)
physical_seen = set()
with (a.plan / 'embedding_mapping.tsv').open() as f:
    for line in f:
        table, key, rank, physical = map(int, line.split())
        if (table >= len(a.table_lengths) or rank >= a.ranks or physical % a.ranks != rank or
                (table, key) in mapping or (table, physical) in physical_seen):
            raise ValueError('invalid input embedding mapping')
        mapping[table, key] = (rank, physical)
        physical_seen.add((table, physical))
        next_local[table, rank] = max(next_local[table, rank], physical // a.ranks + 1)

offsets = np.cumsum([1 + a.dense] + a.table_lengths)
data = np.memmap(a.input, mode='r', dtype='<u4', shape=(a.input.stat().st_size // (width * 4), width))
additions = []
new_indices = [[] for _ in range(a.ranks)]
for batch in range(a.batches):
    first = start + batch * global_batch
    rows = data[first:first + global_batch]
    for row in rows:
        keys = [(t, int(v)) for t in range(len(a.table_lengths))
                for v in np.unique(row[offsets[t]:offsets[t + 1]])]
        hits = np.zeros(a.ranks, dtype=np.int64)
        for key in keys:
            if key in mapping:
                hits[mapping[key][0]] += 1
        for key in keys:
            if key not in mapping:
                t = key[0]
                rank = min(range(a.ranks), key=lambda r: (-hits[r], next_local[t, r], r))
                physical = int(next_local[t, rank]) * a.ranks + rank
                if physical > np.iinfo(np.uint32).max:
                    raise OverflowError('physical ID exceeds uint32')
                next_local[t, rank] += 1
                mapping[key] = (rank, physical)
                additions.append((t, key[1], rank, physical))
                hits[rank] += 1
    loads = np.zeros(a.ranks, dtype=np.int64)
    for i, row in enumerate(rows):
        hits = np.zeros(a.ranks, dtype=np.int64)
        for t in range(len(a.table_lengths)):
            for value in np.unique(row[offsets[t]:offsets[t + 1]]):
                hits[mapping[t, int(value)][0]] += 1
        candidates = [r for r in range(a.ranks) if loads[r] < a.local_batch]
        rank = min(candidates, key=lambda r: (-hits[r], loads[r], r))
        loads[rank] += 1
        new_indices[rank].append(first + i)
    assert np.all(loads == a.local_batch)

a.out.mkdir()
with (a.out / 'embedding_mapping.tsv').open('w') as out, (a.plan / 'embedding_mapping.tsv').open() as old:
    shutil.copyfileobj(old, out)
    for row in additions:
        out.write('\t'.join(map(str, row)) + '\n')
for r in range(a.ranks):
    np.concatenate((indices[r], np.asarray(new_indices[r], dtype='<u8'))).astype('<u8').tofile(
        a.out / f'sample_indices_{r}.bin')
(a.out / 'manifest.json').write_text(json.dumps(dict(
    input=str(a.input.resolve()), old_plan=str(a.plan.resolve()), ranks=a.ranks,
    local_batch=a.local_batch, old_samples=start, new_batches=a.batches,
    added_ids=len(additions), old_ids_moved=0,
    algorithm='greedy locality and per-table rank load; distinct from offline pricing'), indent=2) + '\n')
if a.publish:
    temporary = a.publish.with_name(a.publish.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as f:
        f.write(str(a.out.resolve()) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, a.publish)
print(f'Appended {a.batches} batches and {len(additions)} IDs; old owners unchanged')
