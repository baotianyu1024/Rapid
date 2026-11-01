"""Delete only inventoried, regenerable experiment data; retain source inputs/code."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path('/mnt/hugui/longjiu.bty')
R = ROOT / 'reproduction'
AUDIT = R / 'dataset_cleanup_20261026'
SCOPES = [R / name for name in [
    'rapid_peak_optimization', 'results', 'het_gmp_rapid/results',
    'herald_rapid_gmp/results', 'reviewer_experiments/results',
    'balanced_owner_cache8', 'balanced_owner_cache32',
    'frozen_incremental_20261025', 'rapid_shuffle_incremental_20261025',
    'rapid_day6_diagnosis_20261025', 'rapid_gmp_growth_20261025',
    'metis_seed_day6', 'metis_seed_day6_8192', 'metis_seed_day6_uniform8192']]
PROTECTED = [ROOT / 'data', ROOT / 'store', ROOT / 'code', ROOT / 'paper',
    R / 'reviewer_experiments/data',
    R / 'herald_rapid_gmp/results/default8/latest_rapid',
    R / 'het_gmp_rapid/results/rank64_balance/latest_rapid',
    R / 'het_gmp_rapid/results/rank64_balance/data',
    R / 'het_gmp_rapid/results/main64/data']
for rank in [8, 64]:
    PROTECTED += [R / f'rapid_peak_optimization/r{rank}_guard_budget_original_owners',
                  R / f'rapid_peak_optimization/stream_init_{rank}']
SKIP_DIRS = {'.git', '__pycache__', 'scripts', 'src', 'source', 'latest_source',
             'working', 'build', 'tests', 'test', 'baseline', 'deps', 'toolchain',
             'mamba_root', 'torchrec_compare_env', 'toolchain_packages', 'conda_pkgs'}
CODE_SUFFIXES = {'.py', '.sh', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh',
                 '.rs', '.js', '.ts', '.cmake', '.so'}


def protected(path):
    return any(path == base or base in path.parents for base in PROTECTED)


def signature(path):
    st = path.lstat()
    return dict(device=st.st_dev, inode=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, links=st.st_nlink, blocks=st.st_blocks)


def code_snapshot():
    result = {}
    for base in [ROOT / 'code', R]:
        for directory, dirs, names in os.walk(base):
            dirs[:] = [n for n in dirs if n not in {
                '.git', '__pycache__', 'deps', 'toolchain', 'mamba_root',
                'torchrec_compare_env', 'toolchain_packages', 'conda_pkgs'}]
            for name in names:
                p = Path(directory) / name
                if not p.is_symlink() and (p.suffix in CODE_SUFFIXES or name in {'Makefile', 'CMakeLists.txt'}):
                    result[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def original_snapshot():
    result = {}
    for base in [ROOT / 'data', ROOT / 'store', R / 'reviewer_experiments/data']:
        for directory, _, names in os.walk(base):
            for name in names:
                p = Path(directory) / name
                if not p.is_symlink():
                    s = signature(p)
                    result[str(p)] = {k: s[k] for k in ['device', 'inode', 'size', 'mtime_ns']}
    return result


def plan():
    AUDIT.mkdir(exist_ok=False)
    candidates = []
    for scope in SCOPES:
        for directory, dirs, names in os.walk(scope):
            dirs[:] = [n for n in dirs if n not in SKIP_DIRS
                       and not protected(Path(directory) / n)
                       and not (Path(directory) / n).is_symlink()]
            for name in names:
                p = Path(directory) / name
                if p.is_symlink() or protected(p) or p.resolve() != p:
                    continue
                if p.suffix not in {'.bin', '.npy', '.npz', '.pkl'} and not re.fullmatch(r'table_\d+', name):
                    continue
                s = signature(p)
                # Keep compact traffic matrices/measurement artifacts and tiny fixtures.
                if s['size'] < 1024 * 1024:
                    continue
                with p.open('rb') as stream:
                    magic = stream.read(4)
                if magic.startswith(b'\x7fELF') or magic.startswith(b'#!'):
                    continue
                candidates.append(dict(path=str(p), scope=str(scope), **s))
    counts = Counter((x['device'], x['inode']) for x in candidates)
    unique = {(x['device'], x['inode']): x for x in candidates}
    reclaim = sum(x['blocks'] * 512 for k, x in unique.items() if counts[k] == x['links'])
    groups = Counter()
    for x in unique.values():
        groups[x['scope']] += x['blocks'] * 512
    result = dict(files=candidates, protected_roots=list(map(str, PROTECTED)),
                  estimated_reclaim_bytes=reclaim, unique_allocated_bytes_by_scope=dict(groups),
                  policy='Only >=1MiB generated arrays, binary datasets, table maps and replay checkpoints in explicit experiment scopes. No source files, original inputs, reports, logs, dependencies, executables, or current launcher seed dependencies are removed.')
    (AUDIT / 'plan.json').write_text(json.dumps(result, indent=2))
    (AUDIT / 'code_before.json').write_text(json.dumps(code_snapshot(), indent=2))
    (AUDIT / 'originals_before.json').write_text(json.dumps(original_snapshot(), indent=2))
    print(json.dumps(dict(files=len(candidates), estimated_reclaim_GiB=reclaim/2**30,
                         unique_GiB_by_scope={k: round(v/2**30, 3) for k, v in groups.items()}), indent=2))


def apply():
    data = json.loads((AUDIT / 'plan.json').read_text())
    paths = {x['path'] for x in data['files']}
    # Refuse deletion of data held open by a running experiment.
    busy = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            for fd in (process / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target in paths:
                        busy.append((process.name, target))
                except OSError:
                    pass
        except (PermissionError, FileNotFoundError):
            pass
    if busy:
        raise RuntimeError(f'Generated files in use: {busy[:10]}')
    # Validate the complete inventory before deleting anything.
    for row in data['files']:
        p = Path(row['path'])
        assert not p.is_symlink() and not protected(p) and p.resolve() == p
        assert any(scope in p.parents for scope in SCOPES)
        now = signature(p)
        assert all(now[k] == row[k] for k in ['device', 'inode', 'size', 'mtime_ns']), p
    before = shutil.disk_usage(ROOT)
    with (AUDIT / 'deleted.jsonl').open('x') as log:
        for row in data['files']:
            Path(row['path']).unlink()
            log.write(json.dumps(row) + '\n')
        log.flush()
        os.fsync(log.fileno())
    after = shutil.disk_usage(ROOT)
    assert code_snapshot() == json.loads((AUDIT / 'code_before.json').read_text()), 'Code integrity changed'
    assert original_snapshot() == json.loads((AUDIT / 'originals_before.json').read_text()), 'Original input metadata changed'
    result = dict(success=True, deleted_files=len(data['files']),
                  free_bytes_before=before.free, free_bytes_after=after.free,
                  filesystem_free_increase_bytes=after.free-before.free,
                  estimated_reclaim_bytes=data['estimated_reclaim_bytes'],
                  all_code_sha256_unchanged=True, all_original_inputs_preserved=True,
                  current_launcher_seed_dependencies_preserved=True,
                  note='Historical intermediate datasets/checkpoints require regeneration for replay or resume; code, reports, small traffic arrays, logs and source inputs remain.')
    (AUDIT / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    apply() if args.apply else plan()
