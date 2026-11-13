"""Sample uniformly from the whole downloaded Kaggle training file and run CPS."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

root = Path('/tmp/rapid_kaggle_full_sample_audit')
source = Path('/mnt/nebula/cn-zhangjiakou/juicefs/longjiu.bty/rapid_experiment_data/fresh_datasets/criteo_kaggle/dac/train.txt')
base = Path('/tmp/rapid_fresh_stage/kaggle_8_v2')
num_source = int(Path('/tmp/rapid_kaggle_full_row_count.txt').read_text().split()[0])
n = 1048576
if root.exists():
    raise FileExistsError(root)

def capacity(required, mem_required=8 * 1024**3):
    free = shutil.disk_usage('/tmp').free
    mem = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                   if x.startswith('MemAvailable:'))) * 1024
    if free < required + 8 * 1024**3 or mem < mem_required:
        raise RuntimeError(f'insufficient capacity: {free=}, {mem=}, {required=}, {mem_required=}')

capacity(12 * 1024**3, 96 * 1024**3)
root.mkdir()
selected = np.sort(np.random.default_rng(2026).choice(num_source, size=n, replace=False)).astype(np.uint32)
capacity(selected.nbytes)
np.save(root / 'source_row_indices.npy', selected)
selected_file = root / 'sample.tsv'
pos = 0
buffer = []
with source.open('rb') as src, selected_file.open('xb') as dest:
    for index, line in enumerate(src):
        if index == int(selected[pos]):
            buffer.append(line)
            pos += 1
            if len(buffer) >= 10000 or pos == n:
                capacity(sum(map(len, buffer)))
                dest.writelines(buffer)
                buffer.clear()
                if pos % 100000 == 0 or pos == n:
                    print('sampled', pos, '/', n, 'source row', index, flush=True)
            if pos == n:
                break
assert pos == n
binary = root / 'shuffled_sample.bin'
convert = ['/opt/venv/python3.13.12/bin/python', '/home/longjiu.bty/Rapid/code/Rapid/data_preparation/criteo_singlehot_to_multihot.py',
           '--input', str(selected_file), '--output', str(binary), '--max_rows', str(n), '--shuffle_seed', '42']
capacity(4 * 1024**3, 96 * 1024**3)
with (root / 'convert.log').open('w') as log:
    subprocess.run(convert, check=True, stdout=log, stderr=subprocess.STDOUT)
print('conversion complete', flush=True)

def replace(cmd, changes):
    cmd = list(cmd)
    for key, value in changes.items():
        cmd[cmd.index(key) + 1] = str(value)
    return cmd

hot, placement, scheduled = root / 'hot', root / 'placement', root / 'scheduled'
commands = [
    ('hot', replace(json.loads((base / 'hot.command.json').read_text()), {'--input': binary, '--output': hot})),
    ('placement', replace(json.loads((base / 'placement.command.json').read_text()), {'--input_bin': binary, '--hot_ids_dir': hot, '--output_dir': placement})),
    ('schedule', replace(json.loads((base / 'schedule.command.json').read_text()), {'--input_file': binary, '--table_files_dir': placement, '--output_dir': scheduled})),
]
env = dict(os.environ, OMP_NUM_THREADS='8')
for name, cmd in commands:
    capacity(4 * 1024**3)
    (root / (name + '.command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
    with (root / (name + '.log')).open('w') as log:
        subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(name, 'complete', flush=True)

lengths = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
width, start = 14 + sum(lengths), 14 + sum(lengths[:20])
data = np.memmap(binary, dtype='<u4', mode='r').reshape(16, 8, 8192, width)
ranks = [np.memmap(scheduled / f'dataset_{r}.bin', dtype='<u4', mode='r').reshape(16, 8192, width) for r in range(8)]
counts = {'hash': [], 'cps': []}
for batch in range(16):
    for name, source_ in [('hash', data), ('cps', ranks)]:
        values = []
        for rank in range(8):
            a = (source_[batch, rank] if name == 'hash' else source_[rank][batch])[:, start:start + 100].reshape(-1)
            values.append(int(np.unique(a[a % 8 != rank]).size))
        counts[name].append(values)
    print('counted', batch, flush=True)
meta = dict(source_rows=num_source, selected_rows=n, selection='uniform without replacement over the full source',
            selection_seed=2026, subset_shuffle_seed=42,
            selected_indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(), counts=counts)
(root / 'results.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({k: np.mean(v, axis=1).tolist() for k, v in counts.items()}), flush=True)
