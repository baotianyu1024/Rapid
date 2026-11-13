"""Measure the upstream EMBark auto plan in the shared HugeCTR GPU trainer."""
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
OUT = Path('/tmp/rapid_r1_embark_d32_20261104')
SPLITS = Path('/dev/shm/rapid_rw_holdout_20261103')
COMMON = '/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64'
LIBS = (f'/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:'
        f'/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:{COMMON}')
CONFIG = {
    'kaggle': ('13', '3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1'),
    'terabyte': ('13', '3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1'),
    'taobao': ('1', '1,1,1,1'),
}
MODELS = ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=CONFIG, default=list(CONFIG))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--steps', type=int, default=510)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HUGECTR_BUFFERED_IO='1',
               PYTHONPATH='/tmp/rapid_hugectr_build/lib', LD_LIBRARY_PATH=LIBS,
               LD_PRELOAD='/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
               OMP_NUM_THREADS='8')
    for dataset in args.datasets:
        dense, hotness = CONFIG[dataset]
        split = SPLITS / dataset
        data = split / 'native_train.bin'
        if not data.is_file():
            raise FileNotFoundError(data)
        sizes = Path('/tmp/rapid_rw_e2e_20261103') / dataset / 'native_sizes_verified.json'
        for model in args.models:
            for seed in args.seeds:
                if shutil.disk_usage(OUT).free < 5 * 1024**3:
                    raise OSError('less than 5 GiB free before writing experiment logs')
                log = OUT / f'{dataset}_{model}_seed{seed}_s{args.steps}.log'
                if log.exists() and all(marker in log.read_text(errors='replace')
                                        for marker in ('FIT_SECONDS=', 'FIT_READ_BYTES=', 'EMBARK_PLAN=')) and not re.search(
                                            r'Loss:\s*(?:nan|inf)', log.read_text(errors='replace'), re.I):
                    print(f'SKIP {log.name}', flush=True)
                    continue
                cmd = ['/tmp/rapid_torchrec11/bin/python', '-u',
                       str(ROOT / 'code/hugectr_criteo.py'), '--data', str(data),
                       '--train-samples', '983040', '--dense-count', dense,
                       '--hotness', hotness, '--shape-profile', 'reference',
                       '--embed-dim', '32', '--model', model, '--batch-size', '65536',
                       '--steps', str(args.steps), '--display', '100',
                       '--optimizer', 'adagrad', '--lr', '0.01', '--seed', str(seed),
                       '--sizes-json', str(sizes), '--embark-auto', '--embark-memory-gb', '20']
                with log.open('w') as output:
                    result = subprocess.run(cmd, env=env, stdout=output,
                                            stderr=subprocess.STDOUT, check=False)
                contents = log.read_text(errors='replace')
                if (result.returncode or not all(marker in contents for marker in
                                                ('FIT_SECONDS=', 'FIT_READ_BYTES=', 'EMBARK_PLAN='))
                        or re.search(r'Loss:\s*(?:nan|inf)', contents, re.I)):
                    raise RuntimeError(f'{log.name} failed (exit {result.returncode})')
                print(f'DONE {log.name}', flush=True)


if __name__ == '__main__':
    main()
