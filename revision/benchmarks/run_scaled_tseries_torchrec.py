"""Run the reference-shape, all-row-wise TorchRec T-series matrix."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/tmp/rapid_tseries_scaled_20261112')
CASES = {'t180': (8192, 128), 't200': (32768, 512), 't510': (32768, 512)}
MODELS = ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')
COMMON = '/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--steps', type=int, default=510)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--planner-hbm-gb', type=float, default=24)
    args = parser.parse_args()
    env = dict(os.environ, LD_LIBRARY_PATH=COMMON,
               LD_PRELOAD='/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
               OMP_NUM_THREADS='1')
    for dataset in args.datasets:
        source = args.data_root / dataset / 'torchrec'
        meta = json.loads((source / 'input.json').read_text())
        rows = meta['rows']
        batch = json.loads((args.data_root / dataset / 'manifest.json').read_text())['local_batch']
        if meta['rows'] != rows:
            raise ValueError(f'{dataset}: input row count differs')
        for model in args.models:
            stem = f'torchrec_{model}_d32_n{rows}_b{batch}_{args.steps}'
            result = args.data_root / dataset / f'{stem}.json'
            log = args.data_root / dataset / f'{stem}.log'
            if result.is_file() and json.loads(result.read_text()).get('steps') == args.steps:
                print('SKIP', dataset, model, flush=True)
                continue
            if shutil.disk_usage(args.data_root).free < 8 * 1024**3:
                raise OSError('insufficient disk space for training output and reserve')
            command = ['/tmp/rapid_torchrec11/bin/torchrun', '--standalone',
                       '--nproc-per-node=8', str(ROOT / 'code/torchrec_criteo.py'),
                       '--data', str(source), '--input-mode', 'gpu', '--row-wise-only',
                       '--shape-profile', 'reference', '--embed-dim', '32',
                       '--local-batch', str(batch), '--model', model,
                       '--steps', str(args.steps), '--warmup', '0',
                       '--planner-hbm-gb', str(args.planner_hbm_gb),
                       '--optimizer', 'adagrad', '--lr', '0.01', '--seed', '42',
                       '--out', str(result)]
            print('START', dataset, model, flush=True)
            with log.open('w') as stream:
                subprocess.run(command, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            print('DONE', dataset, model,
                  round(json.loads(result.read_text())['samples_per_second']), flush=True)


if __name__ == '__main__':
    main()
