"""Recreate missing tmpfs splits and rerun paired GPU measurements, preserving old logs."""
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'response_audit_20261108'
DATA = Path('/tmp/rapid_response_audit_20261108/taobao')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', nargs='+', type=int, default=[1, 510])
    p.add_argument('--models', nargs='+', default=['dlrm', 'deepfm'])
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--t200-full-vocab-probe', action='store_true')
    p.add_argument('--output-dir', type=Path, default=OUT)
    p.add_argument('--seed', type=int, default=41)
    args = p.parse_args()
    out = args.output_dir
    DATA.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DATA).free < 2**30:
        raise OSError('need 1 GiB for fresh splits and reserve')
    split_info = []
    for kind, name in [('native', 'native_compact.bin'), ('materialized', 'materialized.bin')]:
        source = Path('/tmp/rapid_fresh_stage/taobao_controls') / name
        assert source.stat().st_size == 1048576 * 24
        with source.open('rb') as stream:
            for split, rows in [('train', 983040), ('eval', 65536)]:
                target = DATA / f'{kind}_{split}.bin'
                content = stream.read(rows * 24)
                assert len(content) == rows * 24
                if target.exists():
                    assert target.read_bytes() == content
                else:
                    target.write_bytes(content)
                split_info.append(dict(path=str(target), rows=rows,
                                       sha256=hashlib.sha256(content).hexdigest()))
    (out / 'rerun_inputs.json').write_text(json.dumps(split_info, indent=2)+'\n')
    if args.prepare_only:
        return
    common = '/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64'
    libs = ('/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:'
            '/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:' + common)
    env = dict(os.environ, HUGECTR_BUFFERED_IO='1', PYTHONPATH='/tmp/rapid_hugectr_build/lib',
               LD_LIBRARY_PATH=libs, LD_PRELOAD='/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN', OMP_NUM_THREADS='8')
    commands_path = out / 'gpu_commands.json'
    commands = json.loads(commands_path.read_text()) if commands_path.exists() else []
    if args.t200_full_vocab_probe:
        generator = ROOT / 'code/upstream_fresh/HugeCTR/benchmarks/embedding_collection/dataset/200table_100B_hotness20.py'
        groups = next(ast.literal_eval(node.value) for node in ast.parse(generator.read_text()).body
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'dataset_info'
                                                              for t in node.targets))
        sizes = [v for count, hot, v in groups for _ in range(count)]
        hotness = [hot for count, hot, v in groups for _ in range(count)]
        sizes_path = DATA.parent / 't200_full_vocabulary.json'
        sizes_path.write_text(json.dumps(sizes)+'\n')
        manifest = json.loads(Path('/tmp/rapid_official_t200_20261103/manifest.json').read_text())
        # Pre-read the one official shard into page cache before measuring training.
        digest = hashlib.sha256()
        with Path(manifest['output']).open('rb') as stream:
            for block in iter(lambda: stream.read(16*2**20), b''):
                digest.update(block)
        assert digest.hexdigest() == manifest['sha256']
        cmd = ['/tmp/rapid_torchrec11/bin/python', '-u', str(ROOT / 'code/hugectr_criteo.py'),
               '--data', manifest['output'], '--train-samples', '65536', '--dense-count', '13',
               '--hotness', ','.join(map(str, hotness)), '--sizes-json', str(sizes_path),
               '--strategy', 'reduction', '--shape-profile', 'reference', '--embed-dim', '32',
               '--model', 'dcnv2', '--batch-size', '4096', '--steps', '1', '--display', '1',
               '--optimizer', 'adagrad', '--lr', '0.01']
        log = out / 'gpu_t200_full_vocab_d32_reduction_s1.log'
        if log.exists():
            raise FileExistsError(log)
        commands.append(dict(command=cmd, log=str(log), note='Full vocabulary, uniform dim32; official per-table dims differ.'))
        commands_path.write_text(json.dumps(commands, indent=2)+'\n')
        with log.open('w') as stream:
            result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        print('T200_EXIT', result.returncode, flush=True)
        print(log.read_text()[-5000:], flush=True)
        return
    for model in args.models:
        for steps in args.steps:
            for method in ['hugectr_unique', 'rapid']:
                source = 'native' if method == 'hugectr_unique' else 'materialized'
                cmd = ['/tmp/rapid_torchrec11/bin/python', '-u', str(ROOT / 'code/hugectr_criteo.py'),
                       '--data', str(DATA / f'{source}_train.bin'), '--eval-data', str(DATA / f'{source}_eval.bin'),
                       '--train-samples', '983040', '--eval-samples', '65536', '--eval-after-fit',
                       '--eval-logloss', '--zero-embedding-init', '--dense-count', '1', '--hotness', '1,1,1,1',
                       '--shape-profile', 'reference', '--embed-dim', '32', '--model', model,
                       '--batch-size', '65536', '--steps', str(steps), '--display', str(min(100, steps)),
                       '--optimizer', 'adagrad', '--lr', '0.01', '--seed', str(args.seed)]
                if method == 'hugectr_unique':
                    cmd += ['--sizes-json', '/tmp/rapid_rw_e2e_20261103/taobao/native_sizes_verified.json',
                            '--strategy', 'unique']
                else:
                    cmd += ['--plan', '/tmp/rapid_fresh_stage/taobao_final_plan']
                log = out / f'gpu_taobao_{model}_{method}_seed{args.seed}_s{steps}.log'
                if log.exists():
                    raise FileExistsError(log)
                commands.append(dict(command=cmd, log=str(log), env={k: env[k] for k in
                    ['NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'LD_PRELOAD']}))
                commands_path.write_text(json.dumps(commands, indent=2)+'\n')
                print('RUN', log.name, flush=True)
                with log.open('w') as stream:
                    result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    print(log.read_text()[-4000:])
                    sys.exit(result.returncode)
                print('\n'.join(line for line in log.read_text().splitlines() if line.startswith(
                    ('FIT_SECONDS=', 'EVAL_AUC_EXACT=', 'EVAL_LOGLOSS='))), flush=True)


if __name__ == '__main__':
    main()
