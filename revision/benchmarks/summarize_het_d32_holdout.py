"""Collect native HET-GMP DCNv2 runs with the held-out final batch."""
import csv
import json
import re
from pathlib import Path
import numpy as np


ROOT = Path('/tmp/rapid_het_d32_e2e_20261104')
OUT = Path('/home/longjiu.bty/Rapid/het_d32_holdout_results_20261104.csv')
FIELDS = ('dataset', 'model', 'framework', 'optimizer', 'learning_rate', 'local_batch',
          'embedding_dim', 'steps', 'train_seconds', 'samples_per_second',
          'steady_samples_per_second', 'auc', 'validation_loss', 'validation_rows',
          'input_sha256', 'log_path')


def main():
    rows = []
    for dataset in ('terabyte', 'kaggle', 'taobao'):
        tag = 'sgd_510_precise' if dataset == 'taobao' else 'sgd_510_main'
        stem = f'{dataset}_dcnv2_b8192_d32_{tag}'
        torch_data = (Path('/dev/shm/rapid_read_fair/taobao/torch') if dataset == 'taobao'
                      else Path('/dev/shm/rapid_shuffle2026') / dataset / 'torch')
        stage = (Path('/dev/shm/rapid_het_holdout_d32_20261104/taobao') if dataset == 'taobao'
                 else Path('/dev/shm/rapid_het_holdout_d32_main_20261104') / dataset)
        log_path = ROOT / f'{stem}.log'
        metric_path = ROOT / f'{stem}.metrics'
        if not log_path.exists() or not metric_path.exists():
            continue
        log = log_path.read_text()
        metrics = metric_path.read_text()
        match = re.search(r'^EVAL_METRICS=(.+)$', log, re.MULTILINE)
        times = [float(value) for value in re.findall(
            r'^TRAIN \d+:.* time ([0-9.]+)', metrics, re.MULTILINE)]
        if not match or len(times) != 51:
            continue
        evaluation = json.loads(match.group(1))
        if evaluation['step'] != 510:
            raise ValueError(f'invalid evaluation step: {log_path}')
        meta = json.loads((torch_data / 'input.json').read_text())
        width = (torch_data / 'input.bin').stat().st_size // (meta['rows'] * 4)
        raw = np.memmap(torch_data / 'input.bin', dtype=np.int32, mode='r',
                        shape=(meta['rows'], width))
        labels = np.concatenate((np.load(stage / 'train/labels.npy').reshape(-1),
                                 np.load(stage / 'valid/labels.npy').reshape(-1)))
        if not np.array_equal(raw[:, 0], labels):
            raise ValueError(f'HET labels differ from the main TorchRec input: {dataset}')
        seconds = sum(times)
        rows.append(dict(dataset=dataset, model='dcnv2', framework='HET-GMP',
                         optimizer='SGD', learning_rate=0.01, local_batch=8192,
                         embedding_dim=32, steps=510, train_seconds=f'{seconds:.6f}',
                         samples_per_second=f'{510 * 65536 / seconds:.3f}',
                         steady_samples_per_second=f'{480 * 65536 / sum(times[3:]):.3f}',
                         auc=f'{evaluation["auc"]:.9f}',
                         validation_loss=f'{evaluation["loss"]:.9f}',
                         validation_rows=65536, input_sha256=meta['input_sha256'],
                         log_path=str(log_path)))
    with OUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{OUT}: {len(rows)} completed runs')


if __name__ == '__main__':
    main()
