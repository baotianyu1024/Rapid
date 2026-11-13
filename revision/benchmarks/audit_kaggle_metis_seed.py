"""Rerun initial CPS with a documented METIS seed on the same ordered input."""
import json
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np

base = Path('/tmp/rapid_fresh_stage/kaggle_8_v2')
root = Path('/tmp/rapid_kaggle_metis_seeded_audit')
if root.exists():
    raise FileExistsError(root)

def check_capacity():
    free = shutil.disk_usage('/tmp').free
    mem = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                   if x.startswith('MemAvailable:'))) * 1024
    if free < 12 * 1024**3 or mem < 8 * 1024**3:
        raise RuntimeError(f'insufficient capacity: {free=}, {mem=}')

check_capacity()
root.mkdir()
placement, scheduled = root / 'placement', root / 'scheduled'
place_cmd = json.loads((base / 'placement.command.json').read_text())
place_cmd[place_cmd.index('--output_dir') + 1] = str(placement)
place_cmd += ['--seed_assignment', '/tmp/rapid_kaggle_metis_seed_audit/seed_8.txt']
sched_cmd = json.loads((base / 'schedule.command.json').read_text())
sched_cmd[sched_cmd.index('--output_dir') + 1] = str(scheduled)
sched_cmd[sched_cmd.index('--table_files_dir') + 1] = str(placement)
for name, cmd in [('placement', place_cmd), ('schedule', sched_cmd)]:
    check_capacity()
    (root / (name + '.command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
    with (root / (name + '.log')).open('w') as log:
        subprocess.run(cmd, check=True, env=dict(os.environ, OMP_NUM_THREADS='8'), stdout=log, stderr=subprocess.STDOUT)
    print(name, 'complete', flush=True)

lengths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
width, start = 14 + sum(lengths), 14 + sum(lengths[:20])
ranks = [np.memmap(scheduled / f'dataset_{r}.bin', dtype='<u4', mode='r').reshape(16, 8192, width) for r in range(8)]
counts = []
for batch in range(16):
    row = []
    for rank in range(8):
        ids = ranks[rank][batch, :, start:start + 100].reshape(-1)
        row.append(int(np.unique(ids[ids % 8 != rank]).size))
    counts.append(row)
means = np.mean(counts, axis=1)
(root / 'counts.json').write_text(json.dumps({'counts': counts, 'batch_means': means.tolist()}, indent=2) + '\n')
print('first4', means[:4].mean(), 'last4', means[-4:].mean(), 'means', means.tolist(), flush=True)
