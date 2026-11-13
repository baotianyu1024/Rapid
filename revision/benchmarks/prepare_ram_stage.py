"""Copy freshly downloaded experiment inputs to a checked, isolated RAM filesystem."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, default=Path('/dev/shm/rapid_read_fair'))
p.add_argument('--datasets', nargs='+', choices=['kaggle', 'terabyte', 'taobao'],
               default=['kaggle', 'terabyte', 'taobao'])
a = p.parse_args()
if a.out.exists():
    raise RuntimeError(f'{a.out} already exists; use a new staging directory')
a.out.mkdir(parents=True)
root = Path('/tmp/rapid_fresh_stage')
manifest = []

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

def copy(src, dst):
    size = src.stat().st_size
    mem = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith('MemAvailable:'))
    if shutil.disk_usage(a.out).free < size + 8 * 1024**3 or mem < size + 8 * 1024**3:
        raise RuntimeError(f'insufficient tmpfs/RAM space and reserve before writing {dst}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    s, d = sha(src), sha(dst)
    if s != d:
        raise RuntimeError(f'checksum mismatch: {src} -> {dst}')
    manifest.append(dict(source=str(src), target=str(dst), bytes=size, sha256=s))
    print('verified', dst, size, flush=True)

for ds in a.datasets:
    control = root / ('hugectr_controls' if ds == 'kaggle' else
                      'hugectr_terabyte_controls' if ds == 'terabyte' else 'taobao_controls')
    case = root / f'{ds}_8_v2/data'
    torchdir = root / (f'{ds}_8/data' if ds != 'taobao' else 'taobao_8_v2/data')
    target = a.out / ds
    copy(control / 'native_compact.bin', target / 'native.bin')
    copy(control / ('materialized.bin' if ds == 'taobao' else 'rapid_materialized.bin'), target / 'materialized.bin')
    copy(case / 'input.bin', target / 'source.bin')
    for name in ('input.bin', 'input.json', 'sparse.npy'):
        copy(torchdir / name, target / 'torch' / name)
    for name in ('dense.npy', 'labels.npy', 'sparse.npy', 'partition.npz'):
        copy(root / f'het_{ds}' / name, target / 'het' / name)
(a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
