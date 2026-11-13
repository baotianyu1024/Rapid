"""Summarize 30-warmup/80-measure RAM-backed experiments and process I/O."""
import argparse
import datetime as dt
import json
import re
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=['kaggle', 'terabyte', 'taobao'], required=True)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
root = Path('/tmp/rapid_ram_fair') / a.dataset
variants = (['native', 'materialized', 'online'] if a.dataset == 'taobao'
            else ['reduction', 'unique', 'acp_only', 'materialized', 'online'])
result = {}
losses = {}
for name in variants:
    times, io, setup = [], [], []
    for run in (1, 2, 3):
        text = (root / f'{name}_{run}.log').read_text()
        losses[(name, run)] = {int(n): float(v) for n, v in
            re.findall(r'Iter: (\d+) .*?Loss: ([0-9.eE+-]+)', text)}
        moments = {int(n): dt.datetime.strptime(t, '%H:%M:%S.%f') for t, n in
                   re.findall(r'\[HCTR\]\[(\d\d:\d\d:\d\d\.\d+)\].*Iter: (\d+)', text)}
        times.append((moments[110] - moments[30]).total_seconds() / 80)
        def stat(tag, key):
            match = re.search(rf'IO_PROFILE_{tag} {key}: (\d+)', text)
            if not match:
                raise ValueError(f'missing {tag}/{key} in {name}_{run}')
            return int(match.group(1))
        io.append(dict(read_bytes=stat('END', 'read_bytes') - stat('START', 'read_bytes'),
                       rchar=stat('END', 'rchar') - stat('START', 'rchar')))
        match = re.search(r'INPUT_SETUP_SECONDS=([0-9.]+)', text)
        setup.append(float(match.group(1)) if match else None)
    result[name] = dict(step_seconds=times, samples_per_second=[65536/t for t in times],
                        mean_samples_per_second=statistics.mean(65536/t for t in times),
                        process_io_delta=io, input_setup_seconds=setup)
for mode in ['prefetch', 'gpu']:
    times, ios, startup, peak = [], [], [], []
    for run in (1, 2, 3):
        data = json.loads((root / f'torch_{mode}_{run}.json').read_text())
        losses[(f'torch_{mode}', run)] = data['rank0_loss']
        times.append(data['mean_step_seconds'])
        ios.append(data['io_delta_by_rank'])
        startup.append(data['input_startup_by_rank'])
        peak.append(data['peak_allocated_bytes_by_rank'])
    result[f'torch_{mode}'] = dict(step_seconds=times, samples_per_second=[65536/t for t in times],
        mean_samples_per_second=statistics.mean(65536/t for t in times), io_by_run_rank=ios,
        input_startup_by_run_rank=startup, peak_allocated_by_run_rank=peak)
times, ios, startup = [], [], []
for run in (1, 2, 3):
    lines = (root / f'het_{run}.metrics').read_text().splitlines()
    if len(lines) != 11:
        raise ValueError('HET should have 11 blocks of 10 steps')
    times.append(sum(float(re.search(r'time ([0-9.]+)', s).group(1)) for s in lines[3:]) / 80)
    ranks = [json.loads((root / f'het_{run}_io.rank{r}.json').read_text()) for r in range(8)]
    ios.append([r['io_delta'] for r in ranks])
    startup.append([dict(input_load_seconds=r['input_load_seconds'],
                         input_prepare_seconds=r['input_prepare_seconds']) for r in ranks])
result['het'] = dict(step_seconds=times, samples_per_second=[65536/t for t in times],
                     mean_samples_per_second=statistics.mean(65536/t for t in times),
                     io_by_run_rank=ios, input_startup_by_run_rank=startup)
result['_loss_checks'] = {}
for run in (1, 2, 3):
    for left, right in [('materialized', 'online'), ('torch_prefetch', 'torch_gpu')]:
        x, y = losses[(left, run)], losses[(right, run)]
        if isinstance(x, dict):
            keys = sorted(set(x) & set(y))
            diff = [abs(x[k] - y[k]) for k in keys]
        else:
            if len(x) != len(y):
                raise ValueError('mismatched TorchRec loss counts')
            diff = [abs(a - b) for a, b in zip(x, y)]
        result['_loss_checks'][f'{left}_vs_{right}_{run}'] = dict(
            count=len(diff), max_abs=max(diff) if diff else None)
a.output.write_text(json.dumps(result, indent=2) + '\n')
for name, data in result.items():
    if name.startswith('_'):
        continue
    print(name, [round(1000*t, 2) for t in data['step_seconds']],
          [round(v) for v in data['samples_per_second']], round(data['mean_samples_per_second']))
