"""Run HET-GMP's shared-table trainer on the matched reduced T-series shards."""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'code/upstream_fresh/HET-GMP'
DATA = Path('/tmp/rapid_tseries_allmethods_20261113')
CASES = {'t180': 128, 't200': 512, 't510': 512}
MODELS = {'dlrm': 'dlrm_multi', 'dcnv2': 'dcnv2',
          'wide_deep': 'wdl_multi', 'deepfm': 'dfm_multi'}
COMMON = '/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--steps', type=int, default=510)
    parser.add_argument('--store-rate', type=float, default=0.25)
    parser.add_argument('--local-batch', type=int)
    parser.add_argument('--data-root', type=Path, default=DATA)
    args = parser.parse_args()
    if args.steps % 10:
        parser.error('HET-GMP logs require steps divisible by 10')
    for dataset in args.datasets:
        case = args.data_root / dataset
        info = json.loads((case / 'manifest.json').read_text())
        batch = args.local_batch or info['local_batch']
        stage = case / 'het'
        with np.load(stage / 'partition_balanced.npz') as z:
            vocab = len(z['embed_partition'])
            counts = np.bincount(z['data_partition'], minlength=8)
        if vocab != sum(info['sizes']) or not np.array_equal(
                counts, np.full(8, info['rows'] // 8)):
            raise ValueError('HET-GMP vocabulary or sample partition differs from shared input')
        preload = '/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2'
        if os.environ.get('RAPID_HET_ALLOC_TRACE') == '1':
            preload = '/tmp/rapid_cuda_alloc_fail_trace_20261113.so:' + preload
        env = dict(os.environ,
                   LD_PRELOAD=preload,
                   NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
                   RAPID_HET_DATA=str(stage), RAPID_HET_VOCAB=str(vocab),
                   RAPID_HET_HOTNESS=','.join(map(str, info['lengths'])),
                   RAPID_HET_DENSE_DIM='13', RAPID_HET_LR='0.01', RAPID_HET_BIASES='1',
                   PYTHONPATH=f'{BASE}/python:/tmp/rapid_het_build/lib',
                   LD_LIBRARY_PATH='/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:'
                                   '/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:'
                                   f'/tmp/rapid_cudnn8/nvidia/cudnn/lib:{COMMON}',
                   OMP_NUM_THREADS='1')
        for model in args.models:
            stem = f'het_{model}_d32_n{info["rows"]}_b{batch}_{args.steps}'
            metrics = case / f'{stem}.metrics'
            log = case / f'{stem}.log'
            result = case / f'{stem}.json'
            if result.is_file() and json.loads(result.read_text()).get('steps') == args.steps:
                print('SKIP', dataset, model, flush=True)
                continue
            if shutil.disk_usage(case).free < 8 * 1024**3:
                raise OSError('insufficient disk space for HET-GMP logs and reserve')
            command = ['mpirun', '-np', '8', '/tmp/rapid_torchrec11/bin/python', '-u',
                       'examples/hetuctr.py', '--dataset', 'criteo_rapid',
                       '--model', MODELS[model], '--batch_size', str(batch),
                       '--embed_dim', '32', '--bound', '0', '--iter', str(args.steps),
                       '--log_every', '10', '--partition', str(stage / 'partition_balanced.npz'),
                       '--store_rate', str(args.store_rate), '--benchmark',
                       '--output', str(metrics)]
            print('START', dataset, model, flush=True)
            with log.open('w') as stream:
                subprocess.run(command, cwd=BASE, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            text = metrics.read_text()
            times = [float(value) for value in re.findall(r'\btime ([0-9.]+) speed', text)]
            if len(times) != args.steps // 10:
                raise ValueError(f'{metrics}: expected {args.steps // 10} measured windows')
            seconds = sum(times)
            throughput = args.steps * 8 * batch / seconds
            result.write_text(json.dumps(dict(dataset=dataset, model=model, dim=32,
                                              optimizer='SGD 0.01', steps=args.steps,
                                              local_batch=batch, fit_seconds=seconds,
                                              samples_per_second=throughput,
                                              stage=str(stage)), indent=2) + '\n')
            print('DONE', dataset, model, round(throughput), flush=True)


if __name__ == '__main__':
    main()
