"""Split shuffled Criteo traces on a global-batch boundary for held-out AUC."""
import argparse
import shutil
from pathlib import Path


RECORD_BYTES = 912
GLOBAL_BATCH = 65536
TOTAL_ROWS = 1048576
HEADROOM = 8 * 1024**3


def write_slice(source, destination, offset, length):
    free = shutil.disk_usage(destination.parent).free
    if free < length + HEADROOM:
        raise OSError(f'insufficient space for {destination}: need {length + HEADROOM}, free {free}')
    with source.open('rb') as src, destination.open('wb') as dst:
        src.seek(offset)
        remaining = length
        while remaining:
            chunk = src.read(min(8 * 1024**2, remaining))
            if not chunk:
                raise EOFError(f'{source} ended before {offset + length}')
            dst.write(chunk)
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('kaggle', 'terabyte', 'taobao'), required=True)
    args = parser.parse_args()
    root = (Path('/tmp/rapid_fresh_stage/taobao_controls') if args.dataset == 'taobao'
            else Path('/dev/shm/rapid_shuffle2026') / args.dataset)
    out = Path('/dev/shm/rapid_rw_holdout_20261103') / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    record_bytes = 24 if args.dataset == 'taobao' else RECORD_BYTES
    train_bytes = (TOTAL_ROWS - GLOBAL_BATCH) * record_bytes
    eval_bytes = GLOBAL_BATCH * record_bytes
    sources = {'materialized': 'materialized.bin',
               'native': 'native_compact.bin' if args.dataset == 'taobao' else 'native.bin'}
    for kind, filename in sources.items():
        source = root / filename
        if source.stat().st_size != TOTAL_ROWS * record_bytes:
            raise ValueError(f'unexpected source size: {source}')
        for split, offset, length in (('train', 0, train_bytes), ('eval', train_bytes, eval_bytes)):
            destination = out / f'{kind}_{split}.bin'
            if destination.exists():
                if destination.stat().st_size != length:
                    raise ValueError(f'incomplete destination: {destination}')
                continue
            write_slice(source, destination, offset, length)
            print(f'{destination} {length}', flush=True)


if __name__ == '__main__':
    main()
