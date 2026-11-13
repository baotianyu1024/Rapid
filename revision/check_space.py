"""Reject an experiment data write when the destination filesystem lacks room."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument('--bytes', type=int)
    size.add_argument('--gib', type=float)
    parser.add_argument('--reserve-gib', type=float, default=8)
    args = parser.parse_args()
    target = args.output.resolve()
    while not target.exists():
        target = target.parent
    required = args.bytes if args.bytes is not None else int(args.gib * 2**30)
    required += int(args.reserve_gib * 2**30)
    free = shutil.disk_usage(target).free
    print(f'filesystem={target} free={free} required={required}')
    if free < required:
        parser.error('insufficient free space for the next data write and reserve')


if __name__ == '__main__':
    main()
