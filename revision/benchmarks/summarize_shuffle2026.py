"""Summarize three-run shuffled-input throughput and input diagnostics."""
import argparse
import json
import re
import statistics
from pathlib import Path


def huge(path):
    text = path.read_text()
    blocks = {int(step): float(seconds) for step, seconds in re.findall(
        r'Iter: (\d+) Time\(10 iters\): ([0-9.]+)s', text)}
    if not all(step in blocks for step in range(40, 111, 10)):
        raise ValueError(f'Incomplete HugeCTR run: {path}')
    mean_step = statistics.mean(blocks[step] for step in range(40, 111, 10)) / 10
    def io(mark):
        match = re.search(rf'IO_PROFILE_{mark} read_bytes: (\d+)', text)
        if not match:
            raise ValueError(f'Missing I/O marker: {path}')
        return int(match.group(1))
    setup = re.search(r'INPUT_SETUP_SECONDS=([0-9.]+)', text)
    loss = {int(step): float(value) for step, value in re.findall(
        r'Iter: (\d+).*?Loss: ([0-9.eE+-]+)', text)}
    return dict(samples_per_second=65536 / mean_step, step_ms=mean_step * 1000,
                read_bytes=io('END') - io('START'),
                input_setup_seconds=float(setup.group(1)) if setup else None,
                losses=loss)


def gmp(path):
    lines = path.read_text().splitlines()
    times = {int(step): float(seconds) for step, seconds in (
        re.search(r'TRAIN (\d+):.*?time ([0-9.]+)', line).groups() for line in lines)}
    if not all(step in times for step in range(30, 101, 10)):
        raise ValueError(f'Incomplete GMP run: {path}')
    mean_step = statistics.mean(times[step] for step in range(30, 101, 10)) / 10
    prefix = path.with_suffix('')
    io = [json.loads(Path(str(prefix) + f'_io.rank{rank}.json').read_text())
          for rank in range(8)]
    return dict(samples_per_second=65536 / mean_step, step_ms=mean_step * 1000,
                max_rank_read_bytes=max(item['io_delta']['read_bytes'] for item in io))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('kaggle', 'terabyte'))
    args = parser.parse_args()
    root = Path('/tmp/rapid_shuffle2026_results') / args.dataset
    result = {'dataset': args.dataset, 'runs': 3, 'warmup': 30, 'measured_steps': 80}
    huge_modes = ('hugectr_unique', 'hugectr_reduction', 'hugectr_acp',
                  'rapid_materialized', 'rapid_online')
    for mode in huge_modes:
        runs = [huge(root / f'{mode}_{run}.log') for run in (1, 2, 3)]
        result[mode] = dict(mean_samples_per_second=statistics.mean(
            item['samples_per_second'] for item in runs), runs=runs)
    for mode in ('torch_prefetch', 'torch_gpu'):
        runs = [json.loads((root / f'{mode}_{run}.json').read_text()) for run in (1, 2, 3)]
        result[mode] = dict(mean_samples_per_second=statistics.mean(
            item['samples_per_second'] for item in runs),
            samples_per_second=[item['samples_per_second'] for item in runs],
            max_rank_read_bytes=max(rank['read_bytes'] for item in runs
                                    for rank in item['io_delta_by_rank']))
    for mode in ('het_pageable', 'het_gpu'):
        runs = [gmp(root / f'{mode}_{run}.metrics') for run in (1, 2, 3)]
        result[mode] = dict(mean_samples_per_second=statistics.mean(
            item['samples_per_second'] for item in runs), runs=runs)
    checks = {}
    for run in (1, 2, 3):
        left = result['rapid_materialized']['runs'][run-1]['losses']
        right = result['rapid_online']['runs'][run-1]['losses']
        if sorted(left) != sorted(right):
            raise ValueError(f'RAPID loss steps differ in run {run}')
        checks[f'rapid_loss_max_abs_{run}'] = max(abs(left[step] - right[step]) for step in left)
        left = json.loads((root / f'torch_prefetch_{run}.json').read_text())['rank0_loss']
        right = json.loads((root / f'torch_gpu_{run}.json').read_text())['rank0_loss']
        if len(left) != len(right):
            raise ValueError(f'TorchRec loss steps differ in run {run}')
        checks[f'torch_loss_max_abs_{run}'] = max(abs(x-y) for x, y in zip(left, right))
    result['loss_checks'] = checks
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for mode in (*huge_modes, 'torch_prefetch', 'torch_gpu', 'het_pageable', 'het_gpu'):
        print(mode, round(result[mode]['mean_samples_per_second']))
    print(checks)


if __name__ == '__main__':
    main()
