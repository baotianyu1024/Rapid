"""Count TorchRec planner's logical embedding payload for one global batch."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'Rapid/experiments/baselines'))
from torchrec_count import table_bytes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
tables = []
for table in plan['tables']:
    traffic = table_bytes(table, plan['ranks'], plan['local_batch'], 1)
    tables.append(dict(table=table['table'], sharding_type=table['sharding_type'], **traffic))
result = dict(unit='logical bytes per rank per global batch; FP32 dim 32',
              scope='TorchRec embedding payload protocol estimate; not GPU byte counters',
              forward_bytes=sum(t['forward_bytes'] for t in tables),
              backward_bytes=sum(t['backward_bytes'] for t in tables),
              id_bytes=None if any(t['id_bytes'] is None for t in tables) else sum(t['id_bytes'] for t in tables),
              tables=tables)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print('forward bytes', result['forward_bytes'], 'backward bytes', result['backward_bytes'])
