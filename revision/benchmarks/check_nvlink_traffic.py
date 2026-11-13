"""Record NVLink payload counter deltas around a command without resetting counters."""
import argparse
import json
import re
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', required=True, type=Path)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command and a.command[0] == '--' else a.command
if not command:
    p.error('a command is required after --')


def read_counters():
    raw = subprocess.check_output(['nvidia-smi', 'nvlink', '-gt', 'd'], text=True)
    counters, gpu = {}, None
    for line in raw.splitlines():
        m = re.match(r'GPU (\d+):', line)
        if m:
            gpu = int(m.group(1))
        m = re.search(r'Link (\d+): Data (Tx|Rx): (\d+) KiB', line)
        if m and gpu is not None:
            counters[f'{gpu}/{m.group(1)}/{m.group(2)}'] = int(m.group(3))
    if not counters:
        raise RuntimeError('NVLink counters unavailable')
    return counters

before = read_counters()
run = subprocess.run(command)
after = read_counters()
if before.keys() != after.keys():
    raise RuntimeError('NVLink topology changed during test')
deltas = {k: after[k] - before[k] for k in before}
a.output.write_text(json.dumps(dict(command=command, exit_code=run.returncode,
                                    counter_unit='KiB', deltas=deltas,
                                    total_tx_kib=sum(v for k,v in deltas.items() if k.endswith('/Tx')),
                                    total_rx_kib=sum(v for k,v in deltas.items() if k.endswith('/Rx'))),
                               indent=2) + '\n')
if run.returncode:
    raise SystemExit(run.returncode)
