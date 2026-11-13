"""Generate one HugeCTR benchmark shard with the official Zipf construction."""
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPTS = {
    'T180': '180table_70B_hotness80.py',
    'T200': '200table_100B_hotness20.py',
    'T510': '510table_110B_hotness5.py',
}
SOURCE = Path(__file__).resolve().parents[1] / 'frameworks/HugeCTR/benchmarks/embedding_collection/dataset'


def assigned(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                   for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'missing {name}')


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=SCRIPTS)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.part < 100:
        parser.error('part must be in [0, 99]')
    if args.out.exists():
        parser.error('output directory already exists')
    if not args.out.parent.is_dir():
        parser.error('output parent must already exist')

    source = SOURCE / SCRIPTS[args.dataset]
    tree = ast.parse(source.read_text())
    info = assigned(tree, 'dataset_info')
    rows = assigned(tree, 'n')
    tables = sum(count for count, _, _ in info)
    sparse_width = sum(count * hotness for count, hotness, _ in info)
    expected_bytes = rows * (14 + sparse_width) * 4
    free = shutil.disk_usage(args.out.parent).free
    if free < expected_bytes + 8 * 1024**3:
        raise OSError(f'insufficient output space: free={free}, need={expected_bytes + 8 * 1024**3}')
    if available_ram() < max(32 * 1024**3, expected_bytes * 4):
        raise MemoryError('insufficient available RAM for the official generator')

    args.out.mkdir()
    log = args.out / 'generator.log'
    with log.open('w') as stream:
        subprocess.run([sys.executable, str(source), str(args.part), str(args.out)],
                       check=True, stdout=stream, stderr=subprocess.STDOUT)
    output = args.out / f'{source.stem}_synthetic_alpha1.1.bin.part{args.part}'
    if output.stat().st_size != expected_bytes:
        raise ValueError(f'wrong shard size: {output.stat().st_size} != {expected_bytes}')
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    manifest = dict(dataset=args.dataset, part=args.part, rows=rows, tables=tables,
                    sparse_width=sparse_width, bytes=expected_bytes, sha256=digest.hexdigest(),
                    generator=str(source), generator_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    output=str(output))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
