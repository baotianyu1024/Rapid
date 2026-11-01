"""Rebuild profiles for an immutable saved sample plan, without rescheduling."""
import json
import os
from pathlib import Path
import numpy as np
from stream_window import profiles, ptr


def restore_sample_plan(state, source, data, out):
    source, out = Path(source), Path(out)
    out.mkdir(exist_ok=False)
    (out / 'profile_cache').mkdir()
    cfg = json.loads((source / 'config.json').read_text())
    assignment = np.load(source / 'assignment.npy')
    n, g, b = cfg['rows'], cfg['ranks'], cfg['batch_size']
    assert len(assignment) == n
    for k in range(cfg['batches']):
        assert np.array_equal(np.bincount(assignment[k*g*b:(k+1)*g*b], minlength=g),
                              np.full(g, b))
    for name in ['config.json', 'placement.json']:
        (out / name).write_text((source / name).read_text())
    for name in ['assignment.npy', 'predicted_group_counts.npz']:
        os.link(source / name, out / name)
    col = 1 + cfg['dense']
    for t, length in enumerate(cfg['table_lengths']):
        ids = np.load(source / f'{t}.original_ids.npy')
        assert np.array_equal(state.owner[t][ids], np.load(source / f'{t}.owners.npy'))
        assert np.array_equal(state.gid[t][ids], np.load(source / f'{t}.logical_gids.npy'))
        lut = np.full(len(state.owner[t]), -1, np.int32)
        lut[ids] = np.arange(len(ids))
        pointers = np.array([lut.ctypes.data], np.uintp)
        sizes = np.array([len(lut)], np.int64)
        lengths = np.array([length], np.int32)
        record = np.empty((n, 1 + length), np.uint32)
        record[:, 0] = 0
        record[:, 1:] = data[:, col:col+length]
        profile = np.zeros((len(ids), g), np.uint64)
        profiles(ptr(record), n, g, b, record.shape[1], 0, 1,
                 *map(ptr, [lengths, pointers, sizes, assignment, profile]))
        dtype = np.uint8 if cfg['batches'] <= 8 else np.uint64
        np.save(out / f'profile_cache/{t}.profiles.npy', profile.astype(dtype))
        np.save(out / f'profile_cache/{t}.ids.npy', state.physical[t][ids])
        for suffix in ['original_ids', 'owners', 'logical_gids', 'new_hot_ids']:
            path = source / f'{t}.{suffix}.npy'
            if path.exists():
                os.link(path, out / path.name)
        col += length
    placement = json.loads((out / 'placement.json').read_text())
    placement['sample_plan_restored_without_rescheduling'] = True
    return placement
