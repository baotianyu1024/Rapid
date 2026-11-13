"""Write matched reduced T-series samples as BagPipe ragged JSONL."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/tmp/rapid_tseries_b32_20261113')
OUT = Path('/dev/shm/rapid_bagpipe_tseries_b32_20261113')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('t180', 't200', 't510'))
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--out-root', type=Path, default=OUT)
    args = parser.parse_args()
    case = args.data_root / args.dataset
    info = json.loads((case / 'manifest.json').read_text())
    target = args.out_root / args.dataset
    if target.exists():
        parser.error(f'output already exists: {target}')
    parent = args.out_root if args.out_root.exists() else args.out_root.parent
    if shutil.disk_usage(parent).free < info['input_bytes'] * 4 + 2 * 1024**3:
        raise OSError('insufficient space for BagPipe input and reserve')
    args.out_root.mkdir(exist_ok=True)
    command = ['/tmp/rapid_torchrec11/bin/python', str(ROOT / 'code/export_bagpipe_jsonl.py'),
               '--input', str(case / 'input.bin'), '--output-dir', str(target),
               '--dense-count', '13', '--hotness', ','.join(map(str, info['lengths']))]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
