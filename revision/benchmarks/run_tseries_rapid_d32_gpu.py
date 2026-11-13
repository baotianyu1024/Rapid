"""Train HugeCTR with the exported RAPID CPS/ACP runtime plan for one T shard."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
DATA_ROOT = Path('/tmp/rapid_tseries_d32_20261104')
OUT = Path('/tmp/rapid_tseries_d32_gpu_20261104')
COMMON = '/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('t180', 't200', 't510'))
    parser.add_argument('--steps', type=int, default=510)
    parser.add_argument('--data-override', type=Path,
                        help='Byte-identical input copy on a faster filesystem')
    args = parser.parse_args()
    case = DATA_ROOT / ('t510_verified' if args.dataset == 't510' else args.dataset)
    meta = json.loads((case / 'data/input.json').read_text())
    data = case / 'data/input.bin'
    if args.data_override:
        official = json.loads((case / 'data/input.json').read_text())
        digest = hashlib.sha256()
        with args.data_override.open('rb') as stream:
            for chunk in iter(lambda: stream.read(64 * 1024**2), b''):
                digest.update(chunk)
        if digest.hexdigest() != official['input_sha256']:
            raise ValueError('RAM input differs from the audited official shard')
        data = args.data_override
    plan = case / 'runtime_plan'
    if not (plan / 'manifest.json').exists():
        raise FileNotFoundError(f'Export the complete runtime plan first: {plan}')
    stem = f'{args.dataset}_rapid_adagrad_d32_b512_{args.steps}'
    if args.data_override:
        stem += '_shm'
    if os.environ.get('RAPID_CACHE_MAPPED_BATCHES') == '1':
        stem += '_cache'
    hotness = ','.join(map(str, meta['lengths']))
    env = dict(os.environ, HUGECTR_BUFFERED_IO='1', RAPID_IO_PROFILE='1',
               LD_PRELOAD='/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
               RAPID_PLAN_SOURCE=str(data), RAPID_PLAN_DIR=str(plan),
               RAPID_DENSE_COUNT='13', RAPID_TABLE_LENGTHS=hotness,
               RAPID_MAPPING_THREADS='8', PYTHONPATH='/tmp/rapid_hugectr_build/lib',
               LD_LIBRARY_PATH='/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:'
                               '/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:'
                               f'/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:{COMMON}',
               OMP_NUM_THREADS='8')
    command = ['/tmp/rapid_torchrec11/bin/python', '-u', str(ROOT / 'code/hugectr_criteo.py'),
               '--data', str(data), '--train-samples', '65536', '--dense-count', '13',
               '--hotness', hotness, '--plan', str(plan), '--acp-plan', str(plan / 'acp.json'),
               '--shape-profile', 'reference', '--embed-dim', '32', '--model', 'dcnv2',
               '--batch-size', '4096', '--steps', str(args.steps), '--display', '10',
               '--optimizer', 'adagrad', '--lr', '0.01']
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f'{stem}.log').open('w') as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f'completed {stem}')


if __name__ == '__main__':
    main()
