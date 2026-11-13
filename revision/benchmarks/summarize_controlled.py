"""Summarize controlled warm-cache experiments without discarding outliers."""
import argparse
import datetime as dt
import json
import re
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=('kaggle', 'terabyte', 'taobao'), required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path('/tmp') / f'rapid_{a.dataset}_controlled'
variants = (['native', 'materialized', 'online', 'torchrec', 'het'] if a.dataset == 'taobao'
            else ['reduction', 'unique', 'acp_only', 'materialized', 'online', 'torchrec', 'het'])
result = {}
for variant in variants:
    values = []
    for i in range(1, 4):
        if variant == 'torchrec':
            record = json.loads((root / f'torchrec_{i}.json').read_text())
            seconds = record['mean_step_seconds']
        elif variant == 'het':
            lines = (root / f'het_{i}.metrics').read_text().splitlines()
            if len(lines) != 9:
                raise ValueError(f'expected 9 HET time blocks in {root}/het_{i}.metrics')
            seconds = sum(float(re.search(r'time ([0-9.]+)', line).group(1)) for line in lines[1:]) / 80
        else:
            times = {}
            for line in (root / f'{variant}_{i}.log').read_text().splitlines():
                match = re.search(r'\[HCTR\]\[(\d\d:\d\d:\d\d\.\d+)\].*Iter: (\d+)', line)
                if match:
                    times[int(match.group(2))] = dt.datetime.strptime(match.group(1), '%H:%M:%S.%f')
            seconds = (times[90] - times[10]).total_seconds() / 80
        if seconds <= 0:
            raise ValueError('invalid duration')
        values.append(seconds)
    result[variant] = dict(step_seconds=values, samples_per_second=[65536 / t for t in values],
                           mean_samples_per_second=statistics.mean(65536 / t for t in values))
a.output.write_text(json.dumps(result, indent=2) + '\n')
for key, value in result.items():
    print(key, [round(t * 1000, 2) for t in value['step_seconds']],
          [round(t) for t in value['samples_per_second']], round(value['mean_samples_per_second']))
