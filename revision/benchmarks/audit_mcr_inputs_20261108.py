"""Independent MCR input audit, original-column ablation, and sampled exact counts."""
import argparse
from collections import Counter
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'response_audit_20261108'
TEMP = Path('/tmp/rapid_response_audit_20261108/mcr')
PRIOR = Path('/tmp/rapid_fresh_stage/day6_mcr')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples-per-stratum', type=int, default=8)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(TEMP).free < 3 * 2**30:
        raise OSError('need 3 GiB for original-column data and reserve')
    meta = json.loads((PRIOR / 'manifest.json').read_text())
    n = meta['rows']
    data = np.memmap(meta['source'], dtype='<u4', mode='r', shape=(n, 228))
    base = np.array(meta['base'], dtype=np.uint64)
    lengths = meta['lengths']
    cols = np.cumsum([14] + lengths)
    frequency = np.memmap(PRIOR / 'frequency.u32', dtype='<u4', mode='r')
    results = np.fromfile(PRIOR / 'all_targets.u32', dtype='<u4').reshape(-1, 4)
    statistics = []
    for threshold in [10, 20, 50, 100]:
        chosen = results[results[:, 1] >= threshold]
        high = chosen[:, 2].astype(np.uint64)*100 > chosen[:, 1].astype(np.uint64)*95
        target_tables = np.searchsorted(base, chosen[:, 0], side='right')-1
        partner_tables = np.searchsorted(base, chosen[:, 3], side='right')-1
        statistics.append(dict(min_support=threshold, eligible=len(chosen), high=int(high.sum()),
                               percentage=float(high.mean()*100),
                               high_with_recorded_same_table_witness=int(np.count_nonzero(high & (target_tables == partner_tables)))))
    # Independently scan the unexpanded first column of every table.
    compact = np.lib.format.open_memmap(TEMP / 'original_compact.npy', mode='w+', dtype='<u4', shape=(n, 26))
    freq_parts, original_ids, original_bases = [], [], [0]
    for t in range(26):
        ids, inverse, counts = np.unique(data[:, cols[t]], return_inverse=True, return_counts=True)
        compact[:, t] = inverse.astype(np.uint32) + original_bases[-1]
        freq_parts.append(counts.astype('<u4'))
        original_ids.append(ids)
        original_bases.append(original_bases[-1]+len(ids))
        print('ORIGINAL_COLUMN', t, 'active', len(ids), flush=True)
    compact.flush()
    compact.tofile(TEMP / 'original_rows.u32')
    f_original = np.concatenate(freq_parts)
    f_original.tofile(TEMP / 'original_frequency.u32')
    np.array(original_bases, dtype='<u8').tofile(TEMP / 'original_offsets.u64')
    np.concatenate(original_ids).tofile(TEMP / 'original_ids.u32')
    # New random targets, stratified by support; no claim of population estimation.
    rng = np.random.default_rng(20261108)
    sampled = []
    for lo, hi in [(10, 20), (20, 50), (50, 100), (100, 210)]:
        candidates = np.flatnonzero((results[:, 1] >= lo) & (results[:, 1] < hi))
        sampled.extend(rng.choice(candidates, args.samples_per_stratum, replace=False).tolist())
    exact_checks = []
    for i in sampled:
        u, support, expected, witness = map(int, results[i])
        t = int(np.searchsorted(base, u, side='right')-1)
        occurrences = np.flatnonzero(np.any(data[:, cols[t]:cols[t+1]] == u-int(base[t]), axis=1))
        assert len(occurrences) == support
        counts = Counter()
        for row in data[occurrences]:
            for k in range(26):
                ids = np.unique(row[cols[k]:cols[k+1]]).astype(np.uint64) + base[k]
                ids = ids[(frequency[ids] > 0) & (frequency[ids] < n*.0001)]
                counts.update(int(v) for v in ids if int(v) != u)
        best = max(counts.values(), default=0)
        cross = max((c for v, c in counts.items() if not (int(base[t]) <= v < int(base[t+1]))), default=0)
        same = max((c for v, c in counts.items() if int(base[t]) <= v < int(base[t+1])), default=0)
        assert best == expected and counts[witness] == expected
        check = dict(id=u, table=t, support=support, best=best, best_same_table=same, best_cross_table=cross)
        exact_checks.append(check)
        print('EXACT', check, flush=True)
    result = dict(input=meta['source'], rows=n, min_supports=[10,20,50,100],
                  multi_hot_full_statistics=statistics, sampled_exact_checks=exact_checks,
                  original_column_rows_path=str(TEMP / 'original_rows.u32'),
                  original_active_ids=len(f_original),
                  original_eligible_counts={str(k): int(np.count_nonzero((f_original >= k) & (f_original < n*.0001)))
                                            for k in [10,20,50,100]},
                  comparison_note='Same source rows and original ID columns; target universes differ. Sampled checks validate code, not population proportions.')
    (OUT / 'mcr_input_audit.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
