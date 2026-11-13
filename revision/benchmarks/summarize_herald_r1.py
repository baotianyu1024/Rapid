"""Validate all Herald rank summaries and write R1 training throughput."""

import csv
import json
import math
import re
import re
from pathlib import Path


ROOT = Path('/tmp/rapid_herald_r1_20261105')
OUTPUT = Path('/home/longjiu.bty/Rapid/herald_d32_e2e_results_20261105.csv')
RESPONSE = Path('/home/longjiu.bty/Rapid/中文Response.md')
FIELDS = ('dataset', 'model', 'optimizer', 'local_batch', 'embedding_dim',
          'steps', 'rank_count', 'max_rank_seconds', 'samples_per_second',
          'last_loss_min', 'last_loss_max', 'log_dir')


def summarize(directory):
    match = re.fullmatch(
        r'(kaggle|terabyte|taobao)_(dlrm|dcnv2|wide_deep|deepfm)_510steps',
        directory.name)
    if not match:
        raise ValueError(f'not a full Herald R1 run: {directory}')
    dataset, model = match.groups()
    ranks = []
    for rank in range(8):
        record = json.loads((directory / f'rank_{rank}.json').read_text())
        if (record['dataset'] != dataset or record['model'] != model or
                record['rank'] != rank or record['steps'] != 510 or
                record['local_batch'] != 8192 or record['embedding_dim'] != 32 or
                not math.isfinite(record['last_loss']) or
                not math.isfinite(record['train_seconds']) or
                record['train_seconds'] <= 0):
            raise ValueError(f'invalid Herald rank summary: {directory}/rank_{rank}.json')
        ranks.append(record)
    elapsed = max(record['train_seconds'] for record in ranks)
    losses = [record['last_loss'] for record in ranks]
    return dict(dataset=dataset, model=model, optimizer='adagrad_0.01',
                local_batch=8192, embedding_dim=32, steps=510, rank_count=8,
                max_rank_seconds=f'{elapsed:.6f}',
                samples_per_second=f'{8 * 8192 * 510 / elapsed:.3f}',
                last_loss_min=f'{min(losses):.8f}',
                last_loss_max=f'{max(losses):.8f}', log_dir=str(directory))


def main():
    rows = [summarize(directory) for directory in ROOT.iterdir()
            if directory.is_dir() and all(
                (directory / f'rank_{rank}.json').exists() for rank in range(8)) and
            directory.name.endswith('_510steps')]
    rows.sort(key=lambda row: (row['dataset'], row['model']))
    with OUTPUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        names = {'kaggle': 'Kaggle', 'terabyte': 'Terabyte', 'taobao': 'Taobao'}
        models = {'dlrm': 'DLRM', 'dcnv2': 'DLRM-DCNv2',
                  'wide_deep': 'Wide&Deep', 'deepfm': 'DeepFM'}
        document = RESPONSE.read_text()
        lines = document.splitlines(keepends=True)
        for row in rows:
            label = f"{names[row['dataset']]}／{models[row['model']]}"
            matching = [i for i, line in enumerate(lines)
                        if line.startswith(f'| {label} |') and
                        len(line.split('|')) == 10]
            if len(matching) != 1:
                raise ValueError(f'missing or duplicate R1 row: {label}')
            index = matching[0]
            cells = lines[index].split('|')
            value = f"{float(row['samples_per_second']):,.0f}"
            current = cells[6].strip()
            if current not in ('无同配置 GPU 适配', value) and not re.fullmatch(
                    r'\d{1,3}(?:,\d{3})*', current):
                raise ValueError(f'unexpected Herald R1 cell: {label}: {current}')
            cells[6] = f' {value} '
            lines[index] = '|'.join(cells)
        RESPONSE.write_text(''.join(lines))
    print(f'WROTE {OUTPUT} ROWS={len(rows)}')


if __name__ == '__main__':
    main()
