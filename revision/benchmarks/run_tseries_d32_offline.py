"""Replay the official T-series part0 with the current RAPID offline recipe."""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STEMS = {
    't180': '180table_70B_hotness80',
    't200': '200table_100B_hotness20',
    't510': '510table_110B_hotness5',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=STEMS)
    parser.add_argument('--manifest', type=Path,
                        help='Manifest from generate_official_t_shard.py')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stem = STEMS[args.dataset]
    source = ROOT / 'revision/frameworks/HugeCTR/benchmarks/embedding_collection/dataset' / f'{stem}.py'
    tree = ast.parse(source.read_text())
    groups = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Name) and target.id == 'dataset_info'
                      for target in node.targets))
    lengths = [hot for count, hot, _ in groups for _ in range(count)]
    manifest_path = args.manifest or Path(f'/tmp/rapid_official_{args.dataset}_20261103/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    data = Path(manifest['output'])
    if manifest['tables'] != len(lengths) or manifest['generator_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('official generator or table layout changed')
    digest = hashlib.sha256()
    with data.open('rb') as stream:
        for chunk in iter(lambda: stream.read(64 * 1024**2), b''):
            digest.update(chunk)
    if digest.hexdigest() != manifest['sha256']:
        raise ValueError('official input checksum mismatch')
    output = args.out or Path('/tmp/rapid_tseries_d32_20261104') / args.dataset
    command = [sys.executable, str(ROOT / 'scripts/run_offline.py'),
               '--input', str(data), '--out', str(output), '--ranks', '8',
               '--batch-size', '512', '--batches', '16', '--dataset', 'custom',
               '--dense', '13', '--table-lengths', *map(str, lengths)]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
