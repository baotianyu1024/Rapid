"""Recompute TorchRec protocol estimates from the plans used by training."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'Rapid/experiments/baselines'))
from torchrec_count import table_bytes

root = Path('/tmp/rapid_shuffle2026_results')
for dataset in ('kaggle', 'terabyte', 'taobao'):
    directory = root / dataset
    actual = json.loads((directory / 'torch_actual_plan.json').read_text())
    data = (Path('/tmp/rapid_fresh_stage/taobao_8_v2/data') if dataset == 'taobao'
            else Path('/tmp') / f'rapid_shuffle2026_{dataset}/data')
    meta = json.loads((data / 'input.json').read_text())
    sizes = [end - start for start, end in zip(meta['active_id_offsets'], meta['active_id_offsets'][1:])]
    dim = 132 if dataset == 'taobao' else 32
    tables = []
    for index, size in enumerate(sizes):
        entry = actual[f't{index}']
        table = dict(table=index, rows=size, pooling=meta['lengths'][index], dim=dim,
                     sharding_type=entry['sharding_type'], ranks=entry['ranks'],
                     shards=[dict(size=shape) for shape in entry['shard_sizes']])
        tables.append(table)
    if dataset != 'taobao':
        simulated = json.loads((directory / 'torchrec_plan.json').read_text())['tables']
        for left, right in zip(tables, simulated):
            if (left['sharding_type'], left['ranks'], [x['size'] for x in left['shards']]) != (
                right['sharding_type'], right['ranks'], [x['size'] for x in right['shards']]):
                raise ValueError(f'{dataset} table {left["table"]}: simulated and actual plans differ')
    per_table = [dict(table=t['table'], sharding_type=t['sharding_type'],
                      **table_bytes(t, 8, 8192, 1)) for t in tables]
    result = dict(dataset=dataset, dim=dim, scope='logical protocol estimate from actual training plan; not GPU byte counters',
                  unit='bytes per rank per global batch',
                  forward_bytes=sum(t['forward_bytes'] for t in per_table),
                  backward_bytes=sum(t['backward_bytes'] for t in per_table),
                  per_table=per_table)
    (directory / 'torchrec_actual_traffic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(dataset, 'forward MiB', result['forward_bytes']/2**20,
          'backward MiB', result['backward_bytes']/2**20)
