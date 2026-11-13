"""Collect Herald's eight-GPU throughput and explicit failure status on T-series."""

import argparse
import csv
import json
import os
import signal
import subprocess
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/tmp/rapid_tseries_b32_20261113')
RUN = Path('/tmp/rapid_herald_tseries_b32_20261113')
MODELS = ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')
FIELDS = ('dataset', 'model', 'steps', 'local_batch', 'seconds',
          'samples_per_second', 'loss_last_rank0', 'status', 'log')


def run_with_timeout(command, env, log, timeout):
    with log.open('w') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            descendants = psutil.Process(process.pid).children(recursive=True)
            for child in reversed(descendants):
                child.terminate()
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            for child in descendants:
                if child.is_running():
                    child.kill()
            return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=('t180', 't200', 't510'),
                        default=['t180', 't200', 't510'])
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'tseries_b32_herald_matrix_20261113.csv')
    parser.add_argument('--run-root', type=Path, default=RUN)
    parser.add_argument('--init-std', type=float, default=0.01)
    args = parser.parse_args()
    output = args.output
    rows = []
    for dataset in args.datasets:
        for model in args.models:
            work = args.run_root / f'{dataset}_{model}_{args.steps}steps'
            lr = os.environ.get('RAPID_HERALD_LR', '0.01')
            suffix = '' if args.init_std == 0.01 else f'_init{args.init_std:g}'
            if lr != '0.01':
                suffix += f'_lr{lr}'
            log = DATA / f'{dataset}_{model}_herald_{args.steps}{suffix}.log'
            metrics = sorted(work.glob('rank_*.json'))
            probe_oom = False
            if model == 'dlrm' and len(metrics) != 8:
                probe_log = (DATA / 't180_herald_probe.log' if dataset == 't180'
                             else DATA / f'{dataset}_dlrm_herald_10.log')
                if not probe_log.exists() and dataset != 't180':
                    env = dict(os.environ, RAPID_TSERIES_DATA_ROOT=str(DATA),
                               RAPID_HERALD_T_RUN_ROOT=str(args.run_root),
                               RAPID_HERALD_INIT_STD=str(args.init_std))
                    print('PROBE', dataset, model, flush=True)
                    run_with_timeout(['bash', str(ROOT / 'code/probe_herald_tseries.sh'),
                                      dataset, model, '10'], env, probe_log, 180)
                probe_oom = 'out of memory' in probe_log.read_text(errors='replace').lower()
            if len(metrics) != 8:
                if probe_oom:
                    rows.append(dict(dataset=dataset, model=model, steps=args.steps,
                                     local_batch=32, seconds='', samples_per_second='',
                                     loss_last_rank0='', status='oom_probe_10', log=str(probe_log)))
                    with output.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=FIELDS)
                        writer.writeheader()
                        writer.writerows(rows)
                    print('DONE', dataset, model, 'oom_probe_10', flush=True)
                    continue
                env = dict(os.environ, RAPID_TSERIES_DATA_ROOT=str(DATA),
                           RAPID_HERALD_T_RUN_ROOT=str(args.run_root),
                           RAPID_HERALD_INIT_STD=str(args.init_std))
                print('START', dataset, model, flush=True)
                timeout = min(args.timeout, 180) if model == 'dlrm' else args.timeout
                code = run_with_timeout(['bash', str(ROOT / 'code/probe_herald_tseries.sh'),
                                         dataset, model, str(args.steps)], env, log, timeout)
                metrics = sorted(work.glob('rank_*.json'))
            else:
                code = 0
            data = [json.loads(path.read_text()) for path in metrics]
            complete = (len(data) == 8 and sorted(x['rank'] for x in data) == list(range(8))
                        and all(x['steps'] == args.steps and x['local_batch'] == 32
                                for x in data))
            seconds = max(x['train_seconds'] for x in data) if complete else None
            rank0 = next((x for x in data if x['rank'] == 0), None)
            text = log.read_text(errors='replace') if log.exists() else ''
            status = ('ok' if code == 0 and complete else 'oom' if 'out of memory' in text.lower()
                      else 'nonfinite_loss' if 'nonfinite herald training loss' in text.lower()
                      else f'failed_exit_{code}')
            row = dict(dataset=dataset, model=model, steps=args.steps, local_batch=32,
                       seconds='' if seconds is None else f'{seconds:.6f}',
                       samples_per_second='' if seconds is None else round(args.steps * 256 / seconds),
                       loss_last_rank0='' if rank0 is None else rank0['last_loss'],
                       status=status, log=str(log))
            rows.append(row)
            with output.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print('DONE', dataset, model, status, row['samples_per_second'], flush=True)


if __name__ == '__main__':
    main()
