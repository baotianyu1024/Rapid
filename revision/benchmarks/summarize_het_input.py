"""Summarize HET pageable/pinned/GPU input ablations and CPU loader timings."""
import argparse
import json
import re
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=['kaggle', 'terabyte', 'taobao'], required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path('/tmp/rapid_ram_fair') / a.dataset
out = {}
losses = {}
for mode in ('pageable', 'pinned', 'gpu'):
    secs, ios, losses[mode] = [], [], []
    for run in (1, 2, 3):
        lines = (root / f'het_{mode}_{run}.metrics').read_text().splitlines()
        if len(lines) != 11:
            raise ValueError('expected 110 HET steps')
        secs.append(sum(float(re.search(r'time ([0-9.]+)', s).group(1)) for s in lines[3:]) / 80)
        losses[mode].append([float(re.search(r'loss ([0-9.]+)', s).group(1)) for s in lines])
        ios.append([json.loads((root / f'het_{mode}_{run}_io.rank{rank}.json').read_text())['io_delta']
                    for rank in range(8)])
    profile = root / f'het_{mode}_profile'
    loaders = [json.loads(f.read_text()) for f in profile.glob('loader_*.json')]
    tables = [json.loads(f.read_text()) for f in profile.glob('shared_table_*.json')]
    if len(tables) != 8 or len(loaders) != 24:
        raise ValueError(f'incomplete HET input profiles: {mode}, {len(tables)} tables/{len(loaders)} loaders')
    if any(x['calls'] != 80 for x in tables):
        raise ValueError('unexpected number of profiled training calls')
    loader_per_rank = [sum(x['total_ns'] for x in loaders if int(x['rank']) == rank) / 80e6
                       for rank in range(8)]
    def mean_ns(key):
        return statistics.mean(x[key] / x['calls'] / 1e6 for x in tables)
    out[mode] = dict(step_seconds=secs, samples_per_second=[65536/t for t in secs],
        mean_samples_per_second=statistics.mean(65536/t for t in secs), io_by_run_rank=ios,
        loader_ms_by_rank=loader_per_rank, shared_table_wait_ms=mean_ns('wait_ns'),
        shared_table_views_ms=mean_ns('input_views_ns'), shared_table_submit_ms=mean_ns('submit_ns'))
for mode in ('pinned', 'gpu'):
    out[mode]['max_printed_loss_difference_vs_pageable'] = max(
        abs(x-y) for run in range(3) for x, y in zip(losses[mode][run], losses['pageable'][run]))
a.output.write_text(json.dumps(out, indent=2) + '\n')
for mode, d in out.items():
    print(mode, [round(1000*t, 2) for t in d['step_seconds']],
          'mean rate', round(d['mean_samples_per_second']),
          'max physical reads', max(r['read_bytes'] for run in d['io_by_run_rank'] for r in run),
          'loader max ms', round(max(d['loader_ms_by_rank']), 4),
          'shared wait ms', round(d['shared_table_wait_ms'], 4))
