"""Recover the final offline plan as original-ID and original-row mappings."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'acp'))
from export_hugectr_acp import make_plan

p = argparse.ArgumentParser()
p.add_argument('--case', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
case = a.case.resolve()
if a.out.exists():
    p.error('output must not exist')
cfg = json.loads((case / 'experiment.json').read_text())
g, nb, b = cfg['ranks'], cfg['batches'], cfg['batch_size']
lengths, dense = cfg['table_lengths'], cfg['dense']
width = 1 + dense + sum(lengths)
if shutil.disk_usage(a.out.parent).free < cfg['rows'] * width * 4 + 5 * 1024**3:
    p.error('insufficient output space and reserve')
base = case / 'baseline/rapid_schedule'
final = case / 'optimized/peak_samples'
if not (final / 'complete.json').is_file():
    p.error('final offline plan is incomplete')
a.out.mkdir()
inverse = []
mapping_file = (a.out / 'embedding_mapping.tsv').open('w')
placements = [[[] for _ in range(g)] for _ in lengths]
for rank in range(g):
    with (base / 'placement' / f'table_{rank}').open() as f:
        for line in f:
            table, key = map(int, line.split(':'))
            placements[table][rank].append(key)
for table in range(len(lengths)):
    placement = placements[table]
    lut = np.full(max((len(v) * g for v in placement), default=0), 0xffffffff, np.uint32)
    for rank in range(g):
        keys = np.asarray(sorted(placement[rank]), dtype=np.uint32)
        lut[np.arange(len(keys)) * g + rank] = keys
    for stage in [case / 'baseline/rapid_owner_base', case / 'optimized/symmetric',
                  case / 'optimized/cold_repair', case / 'optimized/hot_guard',
                  case / 'optimized/hot_budget']:
        with np.load(stage / f'table_{table}_remap.npz') as z:
            old, new = z['old'], z['new']
            if len(old) and (old.max() >= len(lut) or np.any(lut[old] == 0xffffffff)):
                raise ValueError(f'unmapped ID in {stage}, table {table}')
            nxt = np.full(int(new.max()) + 1 if len(new) else 0, 0xffffffff, np.uint32)
            if len(np.unique(new)) != len(new):
                raise ValueError('physical ID collision')
            nxt[new] = lut[old]
            lut = nxt
    inverse.append(lut)
    for physical, original in enumerate(lut):
        if original != 0xffffffff:
            mapping_file.write(f'{table}\t{original}\t{physical % g}\t{physical}\n')
mapping_file.close()

raw = np.memmap(case / 'data/input.bin', dtype='<u4', mode='r').reshape(-1, width)
scheduled = [np.memmap(final / 'scheduled' / f'dataset_{r}.bin', dtype='<u4',
                       mode='r').reshape(-1, width) for r in range(g)]
out = [open(a.out / f'sample_indices_{r}.bin', 'wb') for r in range(g)]
offsets = np.cumsum([1 + dense] + lengths)
for batch in range(nb):
    restored = np.concatenate([s[batch*b:(batch+1)*b] for s in scheduled]).copy()
    for t in range(len(lengths)):
        section = restored[:, offsets[t]:offsets[t+1]]
        if section.size and (section.max() >= len(inverse[t]) or
                             np.any(inverse[t][section] == 0xffffffff)):
            raise ValueError(f'unknown physical ID in batch {batch}, table {t}')
        section[:] = inverse[t][section]
    original = np.ascontiguousarray(raw[batch*g*b:(batch+1)*g*b])
    left = original.view(f'V{width*4}').reshape(-1)
    right = np.ascontiguousarray(restored).view(f'V{width*4}').reshape(-1)
    i = np.argsort(left, kind='stable')
    j = np.argsort(right, kind='stable')
    if not np.array_equal(left[i], right[j]):
        raise ValueError(f'complete records differ in global batch {batch}')
    permutation = np.empty(g*b, dtype='<u8')
    permutation[j] = i.astype('<u8') + batch*g*b
    for rank in range(g):
        permutation[rank*b:(rank+1)*b].tofile(out[rank])
for f in out:
    f.close()
profile_path = final / 'cps_metrics.json'
profile_bytes = profile_path.read_bytes()
profile = json.loads(profile_bytes)
if (profile['ranks'], profile['batch_size'], profile['batches']) != (g, b, nb):
    raise ValueError('ACP profile does not match the exported runtime plan')
acp = make_plan(profile, 'remote')
if acp['table_lengths'] != lengths:
    raise ValueError('ACP profile table lengths do not match the runtime plan')
acp['profile_path'] = str(profile_path.resolve())
acp['profile_sha256'] = hashlib.sha256(profile_bytes).hexdigest()
(a.out / 'acp.json').write_text(json.dumps(acp, indent=2) + '\n')
(a.out / 'manifest.json').write_text(json.dumps(dict(
    source_case=str(case), source_input=str(case / 'data/input.bin'), ranks=g,
    local_batch=b, batches=nb, record_bytes=width*4, full_record_audit=True,
    acp_plan='acp.json',
    note='Final offline plan reconstructed from existing intermediates; original input is unchanged.'
), indent=2) + '\n')
print(f'Exported {nb} audited batches and {len(lengths)} table mappings')
