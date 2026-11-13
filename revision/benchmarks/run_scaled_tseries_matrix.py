"""Collect paired T-series model throughput from the shared HugeCTR trainer."""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/tmp/rapid_tseries_scaled_20261112')
CASES = {
    't180': (8192, 128, 'input_small.bin', 'runtime_plan_small'),
    't200': (32768, 512, 'input.bin', 'runtime_plan'),
    't510': (32768, 512, 'input.bin', 'runtime_plan'),
}
MODELS = ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')
METHODS = ('hugectr', 'embark', 'rapid')
FIELDS = ('dataset', 'model', 'method', 'train_samples', 'local_batch', 'dim',
          'steps', 'fit_seconds', 'samples_per_second', 'fit_read_bytes',
          'process_wall_seconds', 'log')


def metric(contents, key):
    values = re.findall(rf'^{key}=([0-9.]+)$', contents, flags=re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f'expected one {key}, found {len(values)}')
    return float(values[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--steps', type=int, default=510)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'tseries_scaled_model_matrix_20261113.csv')
    args = parser.parse_args()
    results = args.output
    rows = []
    for dataset in args.datasets:
        samples, batch, input_name, plan_name = CASES[dataset]
        manifest = json.loads((args.data_root / dataset / 'manifest.json').read_text())
        if manifest['rows'] < samples:
            samples, batch = manifest['rows'], manifest['local_batch']
            input_name, plan_name = 'input.bin', 'runtime_plan'
        if dataset == 't180' and manifest['rows'] == samples:
            input_name, plan_name = 'input.bin', 'runtime_plan'
        suffix = '' if input_name == 'input.bin' else '_input_small'
        for model in args.models:
            for method in args.methods:
                log = args.data_root / dataset / f'{method}_{model}_d32_n{samples}_b{batch}_{args.steps}{suffix}.log'
                if not log.is_file() or 'PROCESS_WALL_SECONDS=' not in log.read_text(errors='replace'):
                    command = [sys.executable, str(ROOT / 'code/run_scaled_tseries_gpu.py'),
                               dataset, method, '--model', model, '--steps', str(args.steps),
                               '--local-batch', str(batch), '--train-samples', str(samples),
                               '--data-name', input_name, '--plan-name', plan_name,
                               '--data-root', str(args.data_root)]
                    print('START', dataset, model, method, flush=True)
                    subprocess.run(command, check=True)
                contents = log.read_text(errors='replace')
                if re.search(r'Loss:\s*(?:nan|inf)', contents, flags=re.IGNORECASE):
                    raise ValueError(f'non-finite loss in {log}')
                fit = metric(contents, 'FIT_SECONDS')
                row = dict(dataset=dataset, model=model, method=method,
                           train_samples=samples, local_batch=batch, dim=32,
                           steps=args.steps, fit_seconds=f'{fit:.6f}',
                           samples_per_second=round(args.steps * 8 * batch / fit),
                           fit_read_bytes=round(metric(contents, 'FIT_READ_BYTES')),
                           process_wall_seconds=f'{metric(contents, "PROCESS_WALL_SECONDS"):.6f}',
                           log=str(log))
                rows.append(row)
                with results.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                print('DONE', dataset, model, method, row['samples_per_second'], flush=True)


if __name__ == '__main__':
    main()
