"""Run eight-GPU training on a generated, scaled-vocabulary T-series shard."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/tmp/rapid_tseries_scaled_20261112')
COMMON = '/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('t180', 't200', 't510'))
    parser.add_argument('method', choices=('hugectr', 'rapid', 'embark'))
    parser.add_argument('--steps', type=int, default=510)
    parser.add_argument('--local-batch', type=int, default=512)
    parser.add_argument('--model', choices=('dlrm', 'dcnv2', 'wide_deep', 'deepfm'), default='dcnv2')
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--train-samples', type=int)
    parser.add_argument('--plan-name', default='runtime_plan')
    parser.add_argument('--data-name', default='input.bin')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    case = args.data_root / args.dataset
    meta = json.loads((case / 'manifest.json').read_text())
    train_samples = args.train_samples or meta['rows']
    if args.local_batch < 1 or train_samples < 1 or train_samples > meta['rows'] or train_samples % (8 * args.local_batch):
        parser.error('training samples must fit the input and divide across eight GPUs')
    data = case / args.data_name
    expected_bytes = train_samples * (14 + sum(meta['lengths'])) * 4
    if data.stat().st_size < expected_bytes:
        raise ValueError('input has fewer records than requested')
    if shutil.disk_usage(case).free < 8 * 1024**3:
        raise OSError('insufficient disk space for training logs and reserve')
    hotness = ','.join(map(str, meta['lengths']))
    env = dict(os.environ, HUGECTR_BUFFERED_IO='1', RAPID_IO_PROFILE='1',
               LD_PRELOAD='/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN',
               RAPID_PLAN_SOURCE=str(data), RAPID_DENSE_COUNT='13',
               RAPID_TABLE_LENGTHS=hotness, RAPID_MAPPING_THREADS='8',
               PYTHONPATH='/tmp/rapid_hugectr_build/lib',
               LD_LIBRARY_PATH='/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:'
                               '/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:'
                               f'/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:{COMMON}',
               OMP_NUM_THREADS='8')
    command = ['/tmp/rapid_torchrec11/bin/python', '-u', str(ROOT / 'code/hugectr_criteo.py'),
               '--data', str(data), '--train-samples', str(train_samples),
               '--dense-count', '13', '--hotness', hotness,
               '--shape-profile', 'reference', '--embed-dim', str(args.embed_dim), '--model', args.model,
               '--batch-size', str(8 * args.local_batch), '--steps', str(args.steps), '--display', '10',
               '--optimizer', 'adagrad', '--lr', '0.01']
    if args.method in ('hugectr', 'embark'):
        command.extend(['--sizes-json', str(case / 'sizes.json'), '--strategy',
                        'acp' if args.method == 'embark' else 'unique'])
        if args.method == 'embark':
            command.extend(['--embark-auto', '--embark-memory-gb', '20'])
    else:
        plan = case / args.plan_name
        if not (plan / 'manifest.json').is_file():
            raise FileNotFoundError(f'RAPID runtime plan missing: {plan}')
        plan_meta = json.loads((plan / 'manifest.json').read_text())
        planned_rows = plan_meta['batches'] * plan_meta['ranks'] * plan_meta['local_batch']
        if (plan_meta['ranks'] != 8 or plan_meta['local_batch'] != args.local_batch
                or planned_rows != train_samples or data.stat().st_size != expected_bytes):
            raise ValueError('RAPID plan, batch size, and physical input length must match')
        env['RAPID_PLAN_DIR'] = str(plan)
        command.extend(['--plan', str(plan), '--acp-plan', str(plan / 'acp.json')])
    data_suffix = '' if args.data_name == 'input.bin' else f'_{Path(args.data_name).stem}'
    log = case / f'{args.method}_{args.model}_d{args.embed_dim}_n{train_samples}_b{args.local_batch}_{args.steps}{data_suffix}.log'
    with log.open('w') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    print(f'{log}: exit={result.returncode}', flush=True)
    result.check_returncode()


if __name__ == '__main__':
    main()
