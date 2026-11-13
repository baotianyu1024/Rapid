"""Derive HugeCTR table capacities from a verified exported RAPID plan."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('plan', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
if not (args.plan / 'manifest.json').is_file():
    parser.error('The plan export is not complete')
sizes = [0] * len(json.loads((args.plan / 'acp.json').read_text())['table_lengths'])
with (args.plan / 'embedding_mapping.tsv').open() as stream:
    for line in stream:
        table, _, _, physical = map(int, line.split())
        sizes[table] = max(sizes[table], physical + 1)
if min(sizes) < 1:
    parser.error('An embedding table is empty')
args.output.write_text(json.dumps(sizes) + '\n')
print(sizes)
