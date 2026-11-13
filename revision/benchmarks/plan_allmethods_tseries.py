"""Build and export RAPID's audited offline plan for a reduced T-series shard."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/tmp/rapid_tseries_allmethods_20261113')
CASES = {'t180': 128, 't200': 512, 't510': 512}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=CASES)
    parser.add_argument('--data-root', type=Path, default=DATA)
    args = parser.parse_args()
    case = args.data_root / args.dataset
    info = json.loads((case / 'manifest.json').read_text())
    batch = info['local_batch']
    rows = info['rows']
    if rows % (8 * batch):
        raise ValueError('input must contain complete global batches')
    offline = case / 'offline'
    plan = case / 'runtime_plan'
    if shutil.disk_usage(case).free < max(8 * 1024**3, 12 * info['input_bytes']):
        raise OSError('insufficient space for offline plan and reserve')
    command = [sys.executable, '-u', str(ROOT / 'code/Rapid/scripts/run_offline.py'),
               '--input', str(case / 'input.bin'), '--out', str(offline),
               '--ranks', '8', '--batch-size', str(batch),
               '--batches', str(rows // (8 * batch)), '--dataset', 'custom',
               '--dense', '13', '--table-lengths', *map(str, info['lengths'])]
    if not (offline / 'complete.json').is_file():
        with (case / 'offline.log').open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           check=True)
    if not (plan / 'manifest.json').is_file():
        if shutil.disk_usage(case).free < info['input_bytes'] + 8 * 1024**3:
            raise OSError('insufficient space for runtime plan and reserve')
        export = [sys.executable, '-u', str(ROOT / 'code/Rapid/experiments/export_runtime_plan.py'),
                  '--case', str(offline), '--out', str(plan)]
        with (case / 'runtime_export.log').open('w') as stream:
            subprocess.run(export, stdout=stream, stderr=subprocess.STDOUT,
                           check=True)
    print(f'{args.dataset}: audited {rows} rows, {len(info["lengths"])} tables', flush=True)


if __name__ == '__main__':
    main()
