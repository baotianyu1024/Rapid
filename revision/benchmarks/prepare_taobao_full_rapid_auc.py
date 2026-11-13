"""Extend the audited Taobao plan to a disjoint full-data AUC split."""

import json
import shutil
from pathlib import Path

import numpy as np


SPLIT = Path('/dev/shm/rapid_taobao_full_auc_20261109')
OLD = Path('/tmp/rapid_fresh_stage/taobao_final_plan')
OUT = SPLIT / 'rapid'
RANKS = 8
LOCAL_BATCH = 8192
GLOBAL_BATCH = RANKS * LOCAL_BATCH
RESERVE = 8 * 1024**3


def check_space(size):
    free = shutil.disk_usage(SPLIT).free
    if free < size + RESERVE:
        raise OSError(f'insufficient /dev/shm capacity: free={free}, required={size + RESERVE}')


def main():
    meta = json.loads((SPLIT / 'manifest.json').read_text())
    sizes = meta['table_sizes']
    train_rows = meta['train_rows']
    eval_rows = meta['eval_rows']
    batches = train_rows // GLOBAL_BATCH
    old_batches = json.loads((OLD / 'manifest.json').read_text())['batches']
    if batches * GLOBAL_BATCH != train_rows or old_batches >= batches or OUT.exists():
        raise ValueError('invalid full-data split or existing RAPID output')
    train = np.memmap(SPLIT / 'train.bin', dtype='<u4', mode='r').reshape(train_rows, 6)
    evaluate = np.memmap(SPLIT / 'eval.bin', dtype='<u4', mode='r').reshape(eval_rows, 6)
    check_space((train_rows + eval_rows) * 24 + train_rows * 8 + sum(sizes) * 48)
    OUT.mkdir()

    owner = [np.full(size, -1, dtype=np.int16) for size in sizes]
    physical = [np.full(size, np.iinfo(np.uint32).max, dtype=np.uint32) for size in sizes]
    next_local = np.zeros((4, RANKS), dtype=np.uint32)
    with (OLD / 'embedding_mapping.tsv').open() as stream:
        for line in stream:
            table, key, rank, pid = map(int, line.split())
            if key >= sizes[table] or owner[table][key] >= 0 or pid % RANKS != rank:
                raise ValueError('invalid initial plan mapping')
            owner[table][key] = rank
            physical[table][key] = pid
            next_local[table, rank] = max(next_local[table, rank], pid // RANKS + 1)

    # New IDs are placed once; the audited initial plan and its physical IDs stay fixed.
    for table, size in enumerate(sizes):
        missing = np.flatnonzero(owner[table] < 0)
        ranks = ((missing.astype(np.uint64) + table) % RANKS).astype(np.int16)
        for rank in range(RANKS):
            ids = missing[ranks == rank]
            local = np.arange(len(ids), dtype=np.uint32) + next_local[table, rank]
            physical[table][ids] = local * RANKS + rank
            owner[table][ids] = rank
            next_local[table, rank] += len(ids)
        if np.any(physical[table] == np.iinfo(np.uint32).max):
            raise ValueError(f'incomplete mapping in table {table}')
        print(f'mapped table {table}: {size} IDs', flush=True)

    mapping = OUT / 'embedding_mapping.tsv'
    check_space(sum(sizes) * 48)
    with mapping.open('x') as stream:
        for table, size in enumerate(sizes):
            for key in range(size):
                stream.write(f'{table}\t{key}\t{int(owner[table][key])}\t{int(physical[table][key])}\n')
    (OUT / 'acp.json').write_bytes((OLD / 'acp.json').read_bytes())

    indices = [np.memmap(OUT / f'sample_indices_{r}.bin', dtype='<u8', mode='w+',
                         shape=(train_rows // RANKS,)) for r in range(RANKS)]
    mapped_train = np.memmap(OUT / 'train.bin', dtype='<u4', mode='w+', shape=train.shape)
    for batch in range(batches):
        start = batch * GLOBAL_BATCH
        rows = train[start:start + GLOBAL_BATCH]
        if batch < old_batches:
            selection = [np.fromfile(OLD / f'sample_indices_{r}.bin', dtype='<u8',
                                     count=LOCAL_BATCH, offset=batch * LOCAL_BATCH * 8) - start
                         for r in range(RANKS)]
        else:
            scores = np.zeros((GLOBAL_BATCH, RANKS), dtype=np.uint8)
            row_ids = np.arange(GLOBAL_BATCH)
            for table in range(4):
                scores[row_ids, owner[table][rows[:, table + 2]]] += 1
            remaining = np.ones(GLOBAL_BATCH, dtype=bool)
            selection = []
            for rank in range(RANKS):
                candidates = np.flatnonzero(remaining)
                if rank == RANKS - 1:
                    chosen = candidates
                else:
                    order = np.argsort(-scores[candidates, rank].astype(np.int16), kind='stable')
                    chosen = candidates[order[:LOCAL_BATCH]]
                remaining[chosen] = False
                selection.append(chosen)
        if any(len(chosen) != LOCAL_BATCH for chosen in selection):
            raise ValueError(f'unbalanced schedule at batch {batch}')
        for rank, chosen in enumerate(selection):
            indices[rank][batch * LOCAL_BATCH:(batch + 1) * LOCAL_BATCH] = start + chosen
            block = rows[chosen].copy()
            for table in range(4):
                block[:, table + 2] = physical[table][block[:, table + 2]]
            mapped_train[start + rank * LOCAL_BATCH:start + (rank + 1) * LOCAL_BATCH] = block
        if batch % 40 == 0:
            print(f'materialized {batch}/{batches} batches', flush=True)
    mapped_train.flush()
    for arr in indices:
        arr.flush()

    check_space(eval_rows * 24)
    mapped_eval = np.memmap(OUT / 'eval.bin', dtype='<u4', mode='w+', shape=evaluate.shape)
    for start in range(0, eval_rows, GLOBAL_BATCH):
        block = evaluate[start:start + GLOBAL_BATCH].copy()
        for table in range(4):
            block[:, table + 2] = physical[table][block[:, table + 2]]
        mapped_eval[start:start + len(block)] = block
    mapped_eval.flush()
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        train_rows=train_rows, eval_rows=eval_rows, batches=batches,
        old_plan_batches=old_batches, old_ids_moved=0,
        full_data_schedule='initial RAPID CPS, then owner-affinity balanced by global batch',
        new_id_placement='static rank-balanced hash',
        acp_selection='static from initial 16 batches',
        validation_ids='mapped to distinct zero-initialized physical rows',
        note='Full-data accuracy comparison; extension is not the original offline pricing planner.'
    ), indent=2) + '\n')
    print(f'prepared {batches} train batches and {eval_rows} eval rows', flush=True)


if __name__ == '__main__':
    main()
