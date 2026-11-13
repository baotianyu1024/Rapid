"""Export a static, per-table HugeCTR ACP plan from a CPS trace profile."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def make_plan(profile, model='remote'):
    ranks, batch = int(profile['ranks']), int(profile['batch_size'])
    tables = profile['per_table']
    if model not in ('paper', 'remote'):
        raise ValueError('unknown ACP cost model')
    if ranks < 2 or batch < 1 or int(profile['batches']) < 1 or not tables:
        raise ValueError('expected at least two ranks, positive batch and tables')
    if sorted(x['table'] for x in tables) != list(range(len(tables))):
        raise ValueError('profile must contain every table exactly once')
    threshold = batch * (ranks if model == 'paper' else ranks - 1)
    result = []
    for item in sorted(tables, key=lambda x: x['table']):
        value, hot = float(item['unique_receive_mean']), int(item['pooling'])
        if not math.isfinite(value) or value < 0 or hot < 1:
            raise ValueError('invalid table statistics')
        result.append(dict(table=item['table'], pooling=hot, unique_remote_vectors=value,
                           reduction_vectors=threshold,
                           strategy='Unique' if value < threshold else 'Reduction'))
    return dict(schema='rapid-hugectr-acp-v1', ranks=ranks, local_batch=batch,
                profile_batches=int(profile['batches']), profile_is_cps=bool(profile.get('cps', False)),
                table_lengths=[x['pooling'] for x in result], cost_model=model,
                selection='static before training; Unique iff mean remote unique < reduction',
                tables=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cost-model', choices=('paper', 'remote'), default='remote',
                        help='remote compares like-for-like network payload; paper retains the legacy mixed-scope threshold')
    args = parser.parse_args()
    raw = args.profile.read_bytes()
    plan = make_plan(json.loads(raw), args.cost_model)
    plan['profile_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(plan, indent=2) + '\n')
    print('Unique:', [x['table'] for x in plan['tables'] if x['strategy'] == 'Unique'])
    print('Reduction:', [x['table'] for x in plan['tables'] if x['strategy'] == 'Reduction'])


if __name__ == '__main__':
    main()
