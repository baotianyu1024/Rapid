"""Matched 64-rank interval-12/16 screen, sharing an audited unmigrated prefix."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT.parents[1])
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='8')
CONTROL = ROOT / 'stream64_phase_control'
PREFIX12 = ROOT / 'stream64_phase_long_exact_prefix12'
PREFIX16 = ROOT / 'stream64_phase_long_exact_prefix16'
REPORT = ROOT / 'long_interval_report'
REPORT.mkdir(exist_ok=True)


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                   check=True, env=ENV)


def spawn(out, interval, stop=0, reuse=False):
    command = [sys.executable, str(ROOT / 'run_stream_phase_pipeline.py'),
               '--ranks', '64', '--windows', '20', '--rows', '2097152',
               '--interval', str(interval), '--fraction', '.001',
               '--seed-schedule', 'grouped', '--refit-new', '--resume', '--out', str(out)]
    if stop:
        command += ['--stop-after', str(stop)]
    if reuse:
        command += ['--reuse-owner-plans', str(CONTROL), '--reuse-sample-plans']
    (out / 'long_interval_command.json').write_text(json.dumps(command, indent=2))
    with out.with_suffix('.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    print('START', proc.pid, out.name, flush=True)
    return proc


def finish(proc, out):
    if proc.wait():
        raise RuntimeError(f'{out} failed; inspect {out.with_suffix(".log")}')
    print('FINISHED', out.name, flush=True)


def audit_prefix(out):
    import numpy as np
    baseline = json.loads((CONTROL / 'metrics.json').read_text())
    metrics = json.loads((out / 'metrics.json').read_text())
    for row, ref in zip(metrics, baseline):
        for key in ['window', 'source_row_start', 'source_row_end', 'training_vectors',
                    'training_metadata_bytes', 'peak_combined_receive', 'phase_bottleneck']:
            assert row[key] == ref[key], (out, key, row['window'])
        folder = f"window_{row['window']:02d}"
        for key in ['fr', 'fs', 'br', 'bs']:
            assert np.array_equal(np.load(out / folder / 'traffic.npz')[key],
                                  np.load(CONTROL / folder / 'traffic.npz')[key])
        assert np.array_equal(np.load(out / folder / 'assignment.npy'),
                              np.load(CONTROL / folder / 'assignment.npy'))
    (out / 'prefix_identity.json').write_text(json.dumps(
        dict(success=True, windows=len(metrics), reference=str(CONTROL)), indent=2))
    print('PREFIX IDENTITY VERIFIED', out.name, len(metrics), flush=True)


sources = ['run_stream_phase_pipeline.py', 'stream_tune.py', 'stream_swaps.cpp',
           'stream_swaps.so', 'sample_group_peak_phase.cpp', 'sample_group_peak_phase.so',
           'stream_owner_state.py', 'stream_window.py', 'stream_alternate.py', 'restore_sample_plan.py']
(REPORT / 'source_manifest.json').write_text(json.dumps(
    {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}, indent=2))
run('clone_stream.py', ROOT / 'stream64_phase_prefix', PREFIX12)
finish(spawn(PREFIX12, 0, stop=12, reuse=True), PREFIX12)
audit_prefix(PREFIX12)
out12 = ROOT / 'stream64_phase_i12_f001'
out16 = ROOT / 'stream64_phase_i16_f001'
run('clone_stream.py', PREFIX12, out12)
run('clone_stream.py', PREFIX12, PREFIX16)
job12 = spawn(out12, 12)
finish(spawn(PREFIX16, 0, stop=16, reuse=True), PREFIX16)
audit_prefix(PREFIX16)
run('clone_stream.py', PREFIX16, out16)
job16 = spawn(out16, 16)
for proc, out in [(job12, out12), (job16, out16)]:
    finish(proc, out)
    run('verify_stream_raw.py', out)
run('compare_streams.py', '--ranks', 64, '--phase', '--out', REPORT)
(REPORT / 'complete.json').write_text(json.dumps(dict(success=True,
    intervals=[12, 16], windows=20, gpu_execution=False), indent=2))
