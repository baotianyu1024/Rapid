"""Audit and merge the matched T-series eight-GPU throughput runs."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/tmp/rapid_tseries_b32_20261113')
DATASETS = ('t180', 't200', 't510')
MODELS = ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')
METHODS = ('hugectr', 'torchrec', 'embark', 'het_gmp', 'herald', 'bagpipe', 'rapid')
FIELDS = ('dataset', 'model', 'method', 'samples_per_second', 'status',
          'steps', 'local_batch', 'dim', 'train_samples', 'source')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    values = {}

    def add(dataset, model, method, speed, status, source):
        key = (dataset, model, method)
        if key in values:
            raise ValueError(f'duplicate result: {key}')
        values[key] = dict(dataset=dataset, model=model, method=method,
                           samples_per_second=speed, status=status,
                           steps=50, local_batch=32, dim=32,
                           train_samples=4096, source=str(source))

    for name in ('tseries_b32_50step_matrix_20261113.csv',
                 'tseries_b32_t180_50step_20261113.csv'):
        source = ROOT / name
        for row in read_csv(source):
            if (int(row['steps']), int(row['local_batch']), int(row['dim']),
                    int(row['train_samples']), int(row['fit_read_bytes'])) != (50, 32, 32, 4096, 0):
                raise ValueError(f'unmatched HugeCTR/EMBark/RAPID run in {source}: {row}')
            add(row['dataset'], row['model'], row['method'],
                row['samples_per_second'], 'ok', row['log'])

    for dataset in DATASETS:
        manifest = json.loads((DATA / dataset / 'manifest.json').read_text())
        if (manifest['rows'], manifest['local_batch']) != (4096, 32):
            raise ValueError(f'unmatched input: {dataset}')
        for model in MODELS:
            for method, stem in (('torchrec', 'torchrec'), ('het_gmp', 'het')):
                source = DATA / dataset / f'{stem}_{model}_d32_n4096_b32_50.json'
                result = json.loads(source.read_text())
                if result['steps'] != 50 or result['local_batch'] != 32:
                    raise ValueError(f'unmatched run: {source}')
                add(dataset, model, method, round(result['samples_per_second']), 'ok', source)

    for method in ('herald', 'bagpipe'):
        source = ROOT / f'tseries_b32_{method}_matrix_20261113.csv'
        for row in read_csv(source):
            if (int(row['steps']), int(row['local_batch'])) != (50, 32):
                raise ValueError(f'unmatched run: {source}')
            status = row['status']
            if method == 'herald' and status.startswith('failed_exit_'):
                log = Path(row['log']).read_text(errors='replace').lower()
                if 'nonfinite herald training loss' in log:
                    status = 'diverged'
            add(row['dataset'], row['model'], method,
                row['samples_per_second'], status, row['log'])

    expected = {(d, m, method) for d in DATASETS for m in MODELS for method in METHODS}
    if set(values) != expected:
        raise ValueError(f'missing={sorted(expected - set(values))}, extra={sorted(set(values) - expected)}')
    output = ROOT / 'tseries_b32_50step_all_methods_20261113.csv'
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for dataset in DATASETS:
            for model in MODELS:
                for method in METHODS:
                    writer.writerow(values[dataset, model, method])
    for dataset in DATASETS:
        for model in MODELS:
            print(dataset, model, *(values[dataset, model, method]['samples_per_second'] or
                                    values[dataset, model, method]['status'] for method in METHODS),
                  sep=' | ')
    print(output)


if __name__ == '__main__':
    main()
