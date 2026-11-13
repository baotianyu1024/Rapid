"""Archive shuffled-input results and runtime plans after checking data-disk capacity."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

base = Path('/tmp')
items = {
    'results': base / 'rapid_shuffle2026_results',
    'kaggle_plan': base / 'rapid_shuffle2026_kaggle_final_plan_fast',
    'terabyte_plan': base / 'rapid_shuffle2026_terabyte_final_plan',
    'report': Path('/home/longjiu.bty/Rapid/全量Shuffle与三框架公平性复测报告.md'),
}
destination = Path('/mnt/nebula/cn-zhangjiakou/juicefs/longjiu.bty/rapid_experiment_data/evidence_20261102/shuffle2026_comparison.tar')
if destination.exists():
    raise FileExistsError(destination)
for name, path in items.items():
    if not path.exists():
        raise FileNotFoundError(f'{name}: {path}')
files = [path for entry in items.values() for path in (entry.rglob('*') if entry.is_dir() else [entry])
         if path.is_file()]
required = sum(path.stat().st_size for path in files) + 8 * 1024**3
destination.parent.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(destination.parent).free
if free < required:
    raise OSError(f'Insufficient data-disk space: need {required:,}, free {free:,} bytes')
with tarfile.open(destination, 'w') as archive:
    for name, entry in items.items():
        archive.add(entry, arcname=name, recursive=True)
digest = hashlib.sha256()
with destination.open('rb') as stream:
    for block in iter(lambda: stream.read(8 << 20), b''):
        digest.update(block)
manifest = dict(archive=str(destination), bytes=destination.stat().st_size,
                sha256=digest.hexdigest(), sources={name: str(path) for name, path in items.items()})
Path(str(destination) + '.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
