"""Train HET-GMP on an official T-series part0 active-ID working set."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path('/home/longjiu.bty/Rapid')
DATA = Path('/dev/shm/rapid_tseries_het_d32_20261104')
OUT = Path('/tmp/rapid_tseries_d32_gpu_20261104')
COMMON = '/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('t180', 't200', 't510'))
    parser.add_argument('--steps', type=int, default=510)
    parser.add_argument('--store-rate', type=float, default=0.25)
    args = parser.parse_args()
    case = 't510_verified' if args.dataset == 't510' else args.dataset
    meta = json.loads((Path('/tmp/rapid_tseries_d32_20261104') / case / 'data/input.json').read_text())
    stage = DATA / args.dataset
    with np.load(stage / 'partition_balanced.npz') as partition:
        vocab = len(partition['embed_partition'])
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f'{args.dataset}_het_dcnv2_sgd_d32_b512_store{args.store_rate:g}_{args.steps}'
    base = ROOT / 'code/upstream_fresh/HET-GMP'
    env = dict(os.environ,
               LD_PRELOAD='/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
               RAPID_HET_DATA=str(stage), RAPID_HET_VOCAB=str(vocab),
               RAPID_HET_HOTNESS=','.join(map(str, meta['lengths'])),
               RAPID_HET_DENSE_DIM='13', RAPID_HET_LR='0.01', RAPID_HET_BIASES='1',
               PYTHONPATH=f'{base}/python:/tmp/rapid_het_build/lib',
               LD_LIBRARY_PATH='/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:'
                               '/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:'
                               f'/tmp/rapid_cudnn8/nvidia/cudnn/lib:{COMMON}',
               OMP_NUM_THREADS='1')
    command = ['mpirun', '-np', '8', '/tmp/rapid_torchrec11/bin/python', '-u',
               'examples/hetuctr.py', '--dataset', 'criteo_rapid', '--model', 'dcnv2',
               '--batch_size', '512', '--embed_dim', '32', '--bound', '0',
               '--iter', str(args.steps), '--log_every', '10',
               '--partition', str(stage / 'partition_balanced.npz'),
               '--store_rate', str(args.store_rate), '--benchmark',
               '--output', str(OUT / f'{stem}.metrics')]
    with (OUT / f'{stem}.log').open('w') as stream:
        subprocess.run(command, cwd=base, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    print(f'completed {stem}')


if __name__ == '__main__':
    main()
