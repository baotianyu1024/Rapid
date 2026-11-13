"""Run paired HugeCTR/RAPID 510-step, dim-32 holdout measurements."""
import argparse
import math
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
OUT = Path('/tmp/rapid_r5_paired_d32_20261104')
SPLITS = Path('/dev/shm/rapid_rw_holdout_20261103')
COMMON = '/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64'
LIBS = (f'/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:'
        f'/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:{COMMON}')
MODELS = ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')
CONFIG = {
    'terabyte': ('13', '3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1'),
    'kaggle': ('13', '3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1'),
    'taobao': ('1', '1,1,1,1'),
}


def complete(log):
    if not log.exists():
        return False
    contents = log.read_text(errors='replace')
    if not re.search(r'^EVAL_METRICS=', contents, re.M):
        return False
    values = []
    for name in ('EVAL_AUC_EXACT', 'EVAL_LOGLOSS', 'FIT_SECONDS'):
        match = re.search(rf'^{name}=(.+)$', contents, re.M)
        if not match:
            return False
        values.append(float(match.group(1)))
    return all(math.isfinite(value) for value in values) and 0 <= values[0] <= 1 and values[1] >= 0 and values[2] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=CONFIG, default=list(CONFIG))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument('--steps', type=int, default=510)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HUGECTR_BUFFERED_IO='1',
               PYTHONPATH='/tmp/rapid_hugectr_build/lib', LD_LIBRARY_PATH=LIBS,
               LD_PRELOAD='/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
               OMP_NUM_THREADS='8')
    failures = []
    for dataset in args.datasets:
        dense, hotness = CONFIG[dataset]
        split = SPLITS / dataset
        if not (split / 'native_train.bin').is_file() or not (split / 'materialized_train.bin').is_file():
            raise FileNotFoundError(f'missing verified shuffle input: {split}')
        for model in args.models:
            for seed in args.seeds:
                for method in ('hugectr_unique', 'rapid'):
                    if shutil.disk_usage(OUT).free < 5 * 1024**3:
                        raise OSError('less than 5 GiB free before writing experiment logs')
                    log = OUT / f'{dataset}_{model}_{method}_seed{seed}_s{args.steps}.log'
                    if complete(log):
                        print(f'SKIP {log.name}', flush=True)
                        continue
                    source = 'native' if method == 'hugectr_unique' else 'materialized'
                    cmd = ['/tmp/rapid_torchrec11/bin/python', '-u',
                           str(ROOT / 'code/hugectr_criteo.py'),
                           '--data', str(split / f'{source}_train.bin'),
                           '--eval-data', str(split / f'{source}_eval.bin'),
                           '--train-samples', '983040', '--eval-samples', '65536',
                           '--eval-after-fit', '--eval-logloss', '--zero-embedding-init',
                           '--dense-count', dense, '--hotness', hotness,
                           '--shape-profile', 'reference', '--embed-dim', '32',
                           '--model', model, '--batch-size', '65536',
                           '--steps', str(args.steps), '--display', '100',
                           '--optimizer', 'adagrad', '--lr', '0.01', '--seed', str(seed)]
                    if method == 'hugectr_unique':
                        sizes = Path('/tmp/rapid_rw_e2e_20261103') / dataset / 'native_sizes_verified.json'
                        cmd += ['--sizes-json', str(sizes), '--strategy', 'unique']
                    elif dataset == 'taobao':
                        plan = Path('/tmp/rapid_fresh_stage/taobao_final_plan')
                        cmd += ['--plan', str(plan), '--acp-plan', str(plan / 'acp.json')]
                    else:
                        plan = Path(f'/tmp/rapid_shuffle2026_{dataset}_final_plan')
                        if dataset == 'kaggle':
                            plan = Path('/tmp/rapid_shuffle2026_kaggle_final_plan_fast')
                        sizes = Path('/tmp/rapid_shuffle2026_results') / dataset / 'plan_sizes.json'
                        cmd += ['--sizes-json', str(sizes), '--acp-plan', str(plan / 'acp.json')]
                    with log.open('w') as output:
                        result = subprocess.run(cmd, env=env, stdout=output,
                                                stderr=subprocess.STDOUT, check=False)
                    if result.returncode or not complete(log):
                        failures.append(log.name)
                        print(f'FAILED {log.name} exit={result.returncode}', flush=True)
                    else:
                        print(f'DONE {log.name}', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} runs failed; see {OUT}')


if __name__ == '__main__':
    main()
