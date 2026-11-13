"""Collect eight-GPU BagPipe throughput on the matched T-series shards."""

import argparse
import csv
import os
import re
import signal
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/tmp/rapid_tseries_b32_20261113')
INPUT = Path('/dev/shm/rapid_bagpipe_tseries_b32_20261113')
RUN = Path('/tmp/rapid_bagpipe_tseries_b32_20261113')
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
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=('t180', 't200', 't510'),
                        default=['t180', 't200', 't510'])
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    output = ROOT / 'tseries_b32_bagpipe_matrix_20261113.csv'
    rows = []
    for dataset in args.datasets:
        for model in args.models:
            work = RUN / f'{dataset}_{model}_8gpu_{args.steps}steps'
            log = DATA / f'{dataset}_{model}_bagpipe_{args.steps}.log'
            metrics = sorted(work.glob('training_worker_*_test_bagpipe.log'))
            if len(metrics) != 8:
                env = dict(os.environ, RAPID_BAGPIPE_TRAINERS='8',
                           RAPID_BAGPIPE_DATASET=dataset, RAPID_BAGPIPE_MODEL=model,
                           RAPID_BAGPIPE_LOCAL_BATCH='32', RAPID_BAGPIPE_STEPS=str(args.steps),
                           RAPID_TSERIES_DATA_ROOT=str(DATA),
                           RAPID_BAGPIPE_INPUT_DIR=str(INPUT / dataset),
                           RAPID_BAGPIPE_RUN_ROOT=str(RUN))
                print('START', dataset, model, flush=True)
                code = run_with_timeout(['bash', str(ROOT / 'code/probe_bagpipe_taobao.sh')],
                                        env, log, args.timeout)
                metrics = sorted(work.glob('training_worker_*_test_bagpipe.log'))
            else:
                code = 0
            times = []
            for path in metrics:
                text = path.read_text(errors='replace')
                matches = re.findall(r'Total Time of (\d+) iterations: ([0-9.]+)ms', text)
                if len(matches) == 1 and int(matches[0][0]) == args.steps:
                    times.append(float(matches[0][1]) / 1000)
            seconds = max(times) if len(times) == 8 else None
            rank0 = next((path for path in metrics if 'training_worker_0_' in path.name), None)
            losses = re.findall(r'Loss ([0-9.eE+-]+)', rank0.read_text()) if rank0 else []
            row = dict(dataset=dataset, model=model, steps=args.steps, local_batch=32,
                       seconds='' if seconds is None else f'{seconds:.6f}',
                       samples_per_second='' if seconds is None else round(args.steps * 256 / seconds),
                       loss_last_rank0=losses[-1] if losses else '',
                       status='ok' if code == 0 and seconds is not None else f'failed_exit_{code}',
                       log=str(log))
            rows.append(row)
            with output.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print('DONE', dataset, model, row['status'], row['samples_per_second'], flush=True)


if __name__ == '__main__':
    main()
