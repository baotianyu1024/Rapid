"""Summarize fresh experiments and independently verify original-column MCR counts."""
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'response_audit_20261108_recheck'
TEMP = Path('/tmp/rapid_response_audit_20261108/mcr')


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    original = np.fromfile(OUT / 'mcr_original_results.u32', dtype='<u4').reshape(-1, 4)
    rows = np.memmap(TEMP / 'original_rows.u32', dtype='<u4', mode='r').reshape(-1, 26)
    freq = np.fromfile(TEMP / 'original_frequency.u32', dtype='<u4')
    offsets = np.fromfile(TEMP / 'original_offsets.u64', dtype='<u8')
    multi = np.fromfile('/tmp/rapid_fresh_stage/day6_mcr/all_targets.u32', dtype='<u4').reshape(-1, 4)
    mcr = []
    for threshold in (10, 20, 50, 100):
        entry = {'min_support': threshold}
        for name, results in [('original_columns', original), ('multi_hot', multi)]:
            selected = results[results[:, 1] >= threshold]
            high = selected[:, 2].astype(np.uint64) * 100 > selected[:, 1].astype(np.uint64) * 95
            entry[name] = dict(eligible=len(selected), high=int(high.sum()), percent=float(100*high.mean()))
        mcr.append(entry)
    # Check independently chosen targets against raw sample rows, rather than C++ postings.
    checks = []
    rng = np.random.default_rng(20261109)
    for lo, hi in [(10, 20), (20, 50), (50, 100), (100, 210)]:
        candidates = np.flatnonzero((original[:, 1] >= lo) & (original[:, 1] < hi))
        for idx in rng.choice(candidates, 4, replace=False):
            u, support, best, witness = map(int, original[idx])
            table = int(np.searchsorted(offsets, u, side='right') - 1)
            occurrences = np.flatnonzero(rows[:, table] == u)
            assert len(occurrences) == support
            counter = Counter(int(v) for v in rows[occurrences].ravel()
                              if int(v) != u and freq[int(v)] < len(rows)*.0001)
            assert max(counter.values(), default=0) == best
            if best:
                assert counter[witness] == best
            checks.append(dict(id=u, support=support, best=best, checked=True))
    save('mcr_comparison.json', dict(rows=len(rows), statistics=mcr, independent_checks=checks,
         scope='Same source rows; first column of each table retains original modulo-vocabulary ID. '
               'Target universes differ; percentages are not paired per-ID effects.'))

    historical = {(r['model'], r['method']): r for r in csv.DictReader(
        (ROOT / 'r5_paired_d32_runs_20261104.csv').open())
        if r['dataset'] == 'taobao' and r['seed'] == '41'}
    gpu = []
    for path in sorted(OUT.glob('gpu_taobao_*.log')):
        match = re.fullmatch(r'gpu_taobao_(dlrm|deepfm)_(hugectr_unique|rapid)_seed41_s(\d+)\.log', path.name)
        if not match:
            continue
        model, method, steps = match.groups()
        log = path.read_text()
        values = {field: float(re.search(rf'^{marker}=(.+)$', log, re.M)[1])
                  for field, marker in [('auc', 'EVAL_AUC_EXACT'), ('logloss', 'EVAL_LOGLOSS'),
                                        ('fit_seconds', 'FIT_SECONDS')]}
        entry = dict(model=model, method=method, steps=int(steps), seed=41,
                     log=str(path.relative_to(ROOT)), **values)
        if int(steps) == 510:
            old = historical[model, method]
            entry['historical_auc'] = float(old['auc'])
            entry['historical_logloss'] = float(old['logloss'])
        gpu.append(entry)
    assert len(gpu) == 8
    save('gpu_recheck_summary.json', gpu)

    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('fig4', ROOT / 'code/Rapid/analysis/compute_unique_vs_reduction.py')
    fig4 = module_from_spec(spec)
    spec.loader.exec_module(fig4)
    input_path = Path('/tmp/rapid_fresh_stage/terabyte_8_v2/data/input.bin')
    unique, reduction = fig4.compute_transfer_volumes(str(input_path), 8, 8192, 1)
    assert np.all(reduction == 8192*7) and reduction.sum() == 1490944
    data = np.memmap(input_path, dtype='<u4', mode='r').reshape(-1, 228)[:65536, 14:]
    legacy, independently_unique = [], []
    for first, last in fig4.TABLE_SLICES:
        owners = data[:, first:last] % 8
        requesting_rank = np.repeat(np.arange(8), 8192)
        legacy.append(sum(np.count_nonzero(np.any(owners == rank, axis=1) & (requesting_rank != rank))
                          for rank in range(8))/8)
        independent = 0
        for rank in range(8):
            keys = np.unique(data[rank*8192:(rank+1)*8192, first:last])
            independent += np.count_nonzero(keys % 8 != rank)
        independently_unique.append(independent/8)
    assert np.array_equal(unique, independently_unique)
    save('figure4_recheck.json', dict(rows=65536, ranks=8, local_batch=8192,
         legacy_nonempty_owner_total=float(sum(legacy)), legacy_per_table=legacy,
         fixed_remote_total=int(reduction.sum()), fixed_remote_per_table=reduction.tolist(),
         unique_per_table=unique.tolist(), includes_local_total=8192*8*26,
         note='One batch on fresh input. Does not reproduce the historical 567K exactly.'))

    memory = []
    dataset_dir = ROOT / 'code/upstream_fresh/HugeCTR/benchmarks/embedding_collection/dataset'
    for name, filename in [('T180', '180table_70B_hotness80.py'),
                           ('T200', '200table_100B_hotness20.py'),
                           ('T510', '510table_110B_hotness5.py')]:
        tree = ast.parse((dataset_dir / filename).read_text())
        groups = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                         and t.id == 'dataset_info' for t in node.targets))
        keys = sum(count*size for count, hotness, size in groups)
        weight = keys*32*4/8/2**30
        memory.append(dict(dataset=name, keys=keys, dim=32, ranks=8,
                           per_rank_weights_gib=weight, weights_adagrad_state_gib=weight*2,
                           three_full_arrays_gib=weight*3,
                           note='Three full arrays assumes dense full-vocabulary gradients. '
                                'Sparse/fused gradient memory is implementation dependent.'))
    save('tseries_memory_recheck.json', memory)
    source = ROOT / '中文Response.md'
    save('provenance.json', dict(response_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         submitted_pdf_sha256=hashlib.sha256((ROOT / 'paper_vldb (2).pdf').read_bytes()).hexdigest(),
         review_sha256=hashlib.sha256((ROOT / 'review.html').read_bytes()).hexdigest()))
    print(json.dumps(dict(mcr=mcr, gpu_runs=len(gpu), independent_mcr_targets=len(checks),
                         figure4_remote_vectors=int(reduction.sum())), indent=2))


if __name__ == '__main__':
    main()
