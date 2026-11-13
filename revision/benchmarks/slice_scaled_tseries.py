"""Materialize a shuffled prefix of an existing generated T-series shard."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=4096)
    parser.add_argument('--local-batch', type=int, default=32)
    args = parser.parse_args()
    info = json.loads((args.source / 'manifest.json').read_text())
    if args.out.exists():
        parser.error('output already exists')
    if args.rows < 1 or args.rows > info['rows'] or args.rows % (8 * args.local_batch):
        parser.error('rows must fit the source and contain complete global batches')
    record_bytes = 4 * (14 + sum(info['lengths']))
    amount = args.rows * record_bytes
    if shutil.disk_usage(args.out.parent).free < amount + 8 * 1024**3:
        raise OSError('insufficient space for sliced dataset and reserve')
    args.out.mkdir()
    digest = hashlib.sha256()
    with (args.source / 'input.bin').open('rb') as source, (args.out / 'input.bin').open('wb') as output:
        remaining = amount
        while remaining:
            chunk = source.read(min(64 * 1024**2, remaining))
            if not chunk:
                raise ValueError('source input ended before requested prefix')
            if shutil.disk_usage(args.out).free < len(chunk) + 8 * 1024**3:
                raise OSError('insufficient space before writing next dataset chunk')
            output.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    info.update(rows=args.rows, local_batch=args.local_batch, input_bytes=amount,
                input_sha256=digest.hexdigest(), parent_input=str(args.source / 'input.bin'),
                parent_input_sha256=info['input_sha256'],
                scope='shuffled prefix of the scaled-vocabulary T-series generator')
    for name, contents in (('manifest.json', json.dumps(info, indent=2) + '\n'),
                           ('sizes.json', json.dumps(info['sizes']) + '\n')):
        if shutil.disk_usage(args.out).free < len(contents.encode()) + 8 * 1024**3:
            raise OSError('insufficient space before writing dataset metadata')
        (args.out / name).write_text(contents)
    print(f'{info["dataset"]}: {args.rows} rows, {amount} bytes, sha256={digest.hexdigest()}')


if __name__ == '__main__':
    main()
