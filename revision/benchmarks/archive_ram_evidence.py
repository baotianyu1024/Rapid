#!/usr/bin/env python3
"""Archive reproducibility evidence after checking target disk and RAM capacity."""
import os
import shutil
import tarfile
from pathlib import Path

dest = Path('/mnt/nebula/cn-zhangjiakou/juicefs/longjiu.bty/rapid_experiment_data/evidence_20261102')
dest.mkdir(parents=True, exist_ok=True)
items = [(Path('/tmp/rapid_ram_fair'), 'rapid_ram_fair'),
         (Path('/dev/shm/rapid_read_fair/manifest.json'), 'manifest.json'),
         (Path('/dev/shm/rapid_read_fair/gmp_balance_audit.json'), 'gmp_balance_audit.json'),
         (Path('/home/longjiu.bty/Rapid/内存驻留输入公平性与ACP实验报告.md'), 'report.md')]
for dataset in ('kaggle', 'terabyte', 'taobao'):
    items.append((Path(f'/dev/shm/rapid_read_fair/{dataset}/het/partition_balanced.npz'),
                  f'partitions/{dataset}.npz'))
    for kind in ('hash', 'paper'):
        items.append((Path(f'/tmp/rapid_{dataset}_{kind}_acp.json'), f'acp/{dataset}_{kind}.json'))
size = sum(p.stat().st_size for path, _ in items for p in ([path] if path.is_file() else path.rglob('*'))
           if p.is_file() and p.suffix != '.sqlite')
free = shutil.disk_usage(dest).free
mem = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith('MemAvailable:'))) * 1024
if free < size + 1024**3 or mem < 8 * 1024**3:
    raise RuntimeError(f'insufficient free disk or memory: need {size}, free {free}, memory {mem}')
output = dest / 'ram_input_comparison_20261102.tar'
if output.exists():
    raise FileExistsError(output)
with tarfile.open(output, 'x') as archive:
    for path, name in items:
        archive.add(path, arcname=name, filter=lambda info: None if info.name.endswith('.sqlite') else info)
print(f'{output} ({os.path.getsize(output)} bytes); free before archive {free}, MemAvailable {mem}')
