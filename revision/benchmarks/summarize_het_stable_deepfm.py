"""Validate the SGD 0.01 HET-GMP DeepFM run with stable BCE and fill R1."""

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path('/tmp/rapid_het_d32_e2e_20261104')
STEM = 'kaggle_deepfm_b8192_d32_sgd_510_stable_bce'
RESPONSE = Path('/home/longjiu.bty/Rapid/中文Response.md')
OUTPUT = Path('/home/longjiu.bty/Rapid/het_kaggle_deepfm_stable_bce_20261105.csv')


def main():
    log = (ROOT / f'{STEM}.log').read_text(errors='replace')
    metrics = (ROOT / f'{STEM}.metrics').read_text(errors='replace')
    evaluations = re.findall(r'^EVAL_METRICS=(.+)$', log, re.M)
    times = [float(value) for value in re.findall(
        r'^TRAIN \d+:.* time ([0-9.]+)', metrics, re.M)]
    losses = [float(value) for value in re.findall(
        r'^TRAIN \d+: loss ([0-9.]+)', metrics, re.M)]
    if len(evaluations) != 1 or len(times) != 51 or len(losses) != 51:
        raise ValueError('incomplete HET-GMP 510-step run')
    evaluation = json.loads(evaluations[0])
    if (evaluation['step'] != 510 or
            not all(math.isfinite(value) for value in
                    times + losses + [evaluation['auc'], evaluation['loss']])):
        raise ValueError('nonfinite HET-GMP training or evaluation')
    seconds = sum(times)
    throughput = 510 * 8 * 8192 / seconds
    row = {'dataset': 'kaggle', 'model': 'deepfm', 'optimizer': 'SGD',
           'learning_rate': 0.01, 'local_batch': 8192, 'embedding_dim': 32,
           'steps': 510, 'train_seconds': f'{seconds:.6f}',
           'samples_per_second': f'{throughput:.3f}',
           'auc': f"{evaluation['auc']:.9f}",
           'validation_loss': f"{evaluation['loss']:.9f}",
           'log_path': str(ROOT / f'{STEM}.log')}
    with OUTPUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    lines = RESPONSE.read_text().splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines)
               if line.startswith('| Kaggle／DeepFM |') and
               len(line.split('|')) == 10]
    if len(matches) != 1:
        raise ValueError('missing Kaggle/DeepFM R1 row')
    cells = lines[matches[0]].split('|')
    if cells[5].strip() != 'NaN（SGD 0.01）':
        raise ValueError(f'unexpected HET-GMP R1 cell: {cells[5]}')
    cells[5] = f' {throughput:,.0f}（稳定 BCE） '
    lines[matches[0]] = '|'.join(cells)
    RESPONSE.write_text(''.join(lines))
    print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
