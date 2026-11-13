"""Train native HugeCTR Unique on a compact active-ID T-series shard."""
import argparse
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
    parser.add_argument('--strategy', choices=('unique', 'hash_acp'), default='unique')
    args = parser.parse_args()
    case = 't510_verified' if args.dataset == 't510' else args.dataset
    meta = json.loads((DATA_ROOT / case / 'data/input.json').read_text())
    data = OUT / f'{args.dataset}_native_compact.bin'
    sizes = OUT / f'{args.dataset}_native_compact_sizes.json'
    if not data.exists() or not sizes.exists():
        raise FileNotFoundError('Run materialize_tseries_native.py first')
    stem = f'{args.dataset}_hugectr_{args.strategy}_adagrad_d32_b512_{args.steps}'
    if os.environ.get('RAPID_RUN_TAG'):
        stem += '_' + os.environ['RAPID_RUN_TAG']
    env = dict(os.environ, HUGECTR_BUFFERED_IO='1',
               LD_PRELOAD='/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
               PYTHONPATH='/tmp/rapid_hugectr_build/lib',
               LD_LIBRARY_PATH='/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:'
                               '/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:'
                               f'/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:{COMMON}',
               OMP_NUM_THREADS='8')
    command = ['/tmp/rapid_torchrec11/bin/python', '-u', str(ROOT / 'code/hugectr_criteo.py'),
               '--data', str(data), '--train-samples', '65536', '--dense-count', '13',
               '--hotness', ','.join(map(str, meta['lengths'])), '--sizes-json', str(sizes),
               '--strategy', 'unique' if args.strategy == 'unique' else 'mixed',
               '--shape-profile', 'reference', '--embed-dim', '32',
               '--model', 'dcnv2', '--batch-size', '4096', '--steps', str(args.steps),
               '--display', '10', '--optimizer', 'adagrad', '--lr', '0.01']
    if args.strategy == 'hash_acp':
        audit = json.loads((Path(f'/tmp/rapid_official_{args.dataset}_20261103') /
                            'hash_acp_audit_b512.json').read_text())
        reduction = [str(table['table']) for table in audit['tables']
                     if table['strategy'] == 'Reduction']
        command.extend(['--reduction-tables', ','.join(reduction)])
    with (OUT / f'{stem}.log').open('w') as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f'completed {stem}')


if __name__ == '__main__':
    main()
