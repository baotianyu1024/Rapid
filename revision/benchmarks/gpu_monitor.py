"""Run one command while sampling total device memory with NVML."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import pynvml

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--log', type=Path, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
if a.command and a.command[0] == '--':
    a.command = a.command[1:]
if not a.command or a.out.exists() or a.log.exists():
    p.error('provide a command and fresh output paths')
pynvml.nvmlInit()
handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
peaks = [0] * len(handles)
start = time.monotonic()
with a.log.open('w') as log:
    proc = subprocess.Popen(a.command, stdout=log, stderr=subprocess.STDOUT)
    while proc.poll() is None:
        for i, handle in enumerate(handles):
            peaks[i] = max(peaks[i], pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        time.sleep(0.2)
elapsed = time.monotonic() - start
pynvml.nvmlShutdown()
a.out.write_text(json.dumps(dict(command=a.command, returncode=proc.returncode,
                                 process_wall_seconds=elapsed, peak_device_bytes=peaks,
                                 sample_interval_seconds=0.2,
                                 note='NVML total device memory; includes context and other processes'),
                            indent=2) + '\n')
raise SystemExit(proc.returncode)
