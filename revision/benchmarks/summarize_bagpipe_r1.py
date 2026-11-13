"""Validate BagPipe rank logs and write single-run R1 training throughput."""

import csv
import math
import re
from pathlib import Path


ROOT = Path('/tmp/rapid_bagpipe_20261105')
OUTPUT = Path('/home/longjiu.bty/Rapid/bagpipe_d32_e2e_results_20261105.csv')
FIELDS = ('dataset', 'model', 'optimizer', 'local_batch', 'embedding_dim',
          'steps', 'rank_count', 'max_rank_seconds', 'samples_per_second',
          'last_loss_min', 'last_loss_max', 'cache_rows_per_rank', 'lookahead',
          'input_mode', 'log_dir')


def summarize(directory):
    dataset, model, gpu_count, step_count = re.fullmatch(
        r'(kaggle|terabyte|taobao)_(dlrm|dcnv2|wide_deep|deepfm)_(8gpu)_(510steps)',
        directory.name).groups()
    steps = int(step_count.removesuffix('steps'))
    seconds = []
    losses = []
    for rank in range(8):
        logs = sorted(directory.glob(f'training_worker_{rank}_*.log'),
                      key=lambda path: path.stat().st_mtime)
        if not logs:
            raise ValueError(f'{directory}: missing rank {rank}')
        content = logs[-1].read_text()
        values = [float(value) for value in re.findall(r'INFO:root:Loss ([^\n]+)', content)]
        duration = re.search(rf'Total Time of {steps} iterations: ([\d.]+)ms', content)
        if len(values) != steps or not duration or not all(map(math.isfinite, values)):
            raise ValueError(f'{logs[-1]}: incomplete or nonfinite training')
        seconds.append(float(duration.group(1)) / 1000)
        losses.append(values[-1])
    elapsed = max(seconds)
    return dict(dataset=dataset, model=model, optimizer='adagrad_0.01',
                local_batch=8192, embedding_dim=32, steps=steps, rank_count=8,
                max_rank_seconds=f'{elapsed:.6f}',
                samples_per_second=f'{8 * 8192 * steps / elapsed:.3f}',
                last_loss_min=f'{min(losses):.8f}', last_loss_max=f'{max(losses):.8f}',
                cache_rows_per_rank=1000000 if dataset == 'taobao' else 16000000,
                lookahead=8, input_mode='preparsed_repeated_global_batches',
                log_dir=str(directory))


def main():
    rows = [summarize(directory) for directory in ROOT.iterdir()
            if directory.is_dir() and re.fullmatch(
                r'(kaggle|terabyte|taobao)_(dlrm|dcnv2|wide_deep|deepfm)_8gpu_510steps',
                directory.name) and all(directory.glob('training_worker_0_*.log'))]
    rows.sort(key=lambda row: (row['dataset'], row['model']))
    with OUTPUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'WROTE {OUTPUT} ROWS={len(rows)}')


if __name__ == '__main__':
    main()
