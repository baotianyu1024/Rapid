"""Add seeds 43 and 44 to the existing all-row-wise TorchRec GPU matrix."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
OUT = Path('/tmp/rapid_r1_torchrec_repeats_d32_20261104')
COMMON = '/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64'
MODELS = ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')
DATA = {
    'kaggle': Path('/dev/shm/rapid_shuffle2026/kaggle/torch'),
    'terabyte': Path('/dev/shm/rapid_shuffle2026/terabyte/torch'),
    'taobao': Path('/dev/shm/rapid_read_fair/taobao/torch'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATA, default=list(DATA))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[43, 44])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=COMMON,
               LD_PRELOAD='/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
               OMP_NUM_THREADS='1')
    for dataset in args.datasets:
        if not (DATA[dataset] / 'input.json').is_file():
            raise FileNotFoundError(DATA[dataset])
        for model in args.models:
            for seed in args.seeds:
                if shutil.disk_usage(OUT).free < 5 * 1024**3:
                    raise OSError('less than 5 GiB free before experiment output')
                stem = f'{dataset}_{model}_seed{seed}'
                result = OUT / f'{stem}.json'
                log = OUT / f'{stem}.log'
                if result.exists() and json.loads(result.read_text()).get('eval_last_batch'):
                    print(f'SKIP {stem}', flush=True)
                    continue
                command = ['/tmp/rapid_torchrec11/bin/torchrun', '--standalone',
                           '--nproc-per-node=8', str(ROOT / 'code/torchrec_criteo.py'),
                           '--data', str(DATA[dataset]), '--input-mode', 'prefetch',
                           '--row-wise-only', '--shape-profile', 'reference',
                           '--embed-dim', '32', '--local-batch', '8192',
                           '--model', model, '--steps', '510', '--warmup', '0',
                           '--optimizer', 'adagrad', '--lr', '0.01', '--seed', str(seed),
                           '--eval-last-batch', '--out', str(result)]
                with log.open('w') as stream:
                    subprocess.run(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                print(f'DONE {stem}', flush=True)


if __name__ == '__main__':
    main()
