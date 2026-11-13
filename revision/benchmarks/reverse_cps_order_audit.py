"""Replan the same global batches in reverse order and measure table 20."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

base = Path('/tmp/rapid_fresh_stage/kaggle_8_v2')
root = Path('/tmp/rapid_kaggle_reverse_cps_audit')
if root.exists():
    raise FileExistsError(root)

def capacity(required):
    disk = shutil.disk_usage('/tmp').free
    mem = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                   if x.startswith('MemAvailable:'))) * 1024
    if disk < required + 8 * 1024**3 or mem < 8 * 1024**3:
        raise RuntimeError(f'insufficient disk/memory: {disk=}, {mem=}, {required=}')

capacity(8 * 1024**3)
root.mkdir()
src = base / 'data/input.bin'
reverse = root / 'reversed.bin'
batch_bytes = 65536 * 912
assert src.stat().st_size == 16 * batch_bytes
with src.open('rb') as inp, reverse.open('xb') as out:
    for batch in range(15, -1, -1):
        capacity(batch_bytes)
        inp.seek(batch * batch_bytes)
        data = inp.read(batch_bytes)
        assert len(data) == batch_bytes
        out.write(data)
        print('copied original batch', batch, flush=True)

def replace(command, changes):
    result = list(command)
    for flag, value in changes.items():
        result[result.index(flag) + 1] = str(value)
    return result

placement = root / 'placement'
scheduled = root / 'scheduled'
place_cmd = replace(json.loads((base / 'placement.command.json').read_text()),
                    {'--input_bin': reverse, '--output_dir': placement})
sched_cmd = replace(json.loads((base / 'schedule.command.json').read_text()),
                    {'--input_file': reverse, '--table_files_dir': placement, '--output_dir': scheduled})
env = dict(os.environ, OMP_NUM_THREADS='8')
for name, cmd in [('placement', place_cmd), ('schedule', sched_cmd)]:
    capacity(4 * 1024**3)
    (root / (name + '.command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
    with (root / (name + '.log')).open('w') as log:
        subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(name, 'complete', flush=True)

lengths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
width = 14 + sum(lengths)
start = 14 + sum(lengths[:20])
results = {}
for name, directory in [('original', base / 'baseline/rapid_schedule/scheduled'), ('reverse', scheduled)]:
    ranks = [np.memmap(directory / f'dataset_{r}.bin', dtype='<u4', mode='r').reshape(16, 8192, width) for r in range(8)]
    result = []
    for batch in range(16):
        vals = []
        for rank in range(8):
            ids = ranks[rank][batch, :, start:start + 100].reshape(-1)
            vals.append(int(np.unique(ids[ids % 8 != rank]).size))
        result.append(vals)
    results[name] = result
    means = np.mean(result, axis=1)
    print(name, 'first4', means[:4].mean(), 'last4', means[-4:].mean(), 'means', means.tolist(), flush=True)
(root / 'counts.json').write_text(json.dumps(results, indent=2) + '\n')
