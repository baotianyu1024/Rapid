"""Independently recount W49-W56 traffic from the fresh day6 trace."""

import json
from pathlib import Path

import numpy as np


TRACE = Path('/tmp/rapid_fresh_stage/day6_5767168.bin')
RUN = Path('/tmp/rapid_fresh_stage/day6_incremental')
OUT = Path('/tmp/rapid_incremental_tail_check_20261109.json')
LENGTHS = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5,
           1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
INITIAL_ROWS = 2_097_152
GLOBAL_BATCH = 65_536
LOCAL_BATCH = 8_192


def main():
    data = np.memmap(TRACE, dtype='<u4', mode='r').reshape(-1, 228)
    summary = json.loads((RUN / 'summary.json').read_text())
    by_window = {row['window']: row for row in summary['results']}
    rows = []
    for window in range(49, 57):
        start = INITIAL_ROWS + (window - 1) * GLOBAL_BATCH
        batch = data[start:start + GLOBAL_BATCH]
        if len(batch) != GLOBAL_BATCH:
            raise ValueError(f'incomplete window {window}')
        one_way = np.zeros(8, dtype=np.int64)
        for rank in range(8):
            records = batch[rank * LOCAL_BATCH:(rank + 1) * LOCAL_BATCH]
            col = 14
            for length in LENGTHS:
                ids = np.unique(records[:, col:col + length])
                one_way[rank] += np.count_nonzero(ids % 8 != rank)
                col += length
        pure = float(2 * one_way.mean())
        with np.load(RUN / f'window_{window:03d}' / 'independent_counts.npz') as counts:
            incremental = float((counts['receive'] + counts['send']).sum(1).mean())
        recorded = by_window[window]['mean_rank_vectors_per_step']
        if incremental != recorded or by_window[window]['old_owner_changes'] != 0:
            raise ValueError(f'incremental count or owner mismatch in window {window}')
        rows.append(dict(window=window, pure_unique_both_directions=pure,
                         pure_unique_one_way=pure / 2,
                         incremental_both_directions=incremental))
    result = dict(source=str(TRACE), windows=rows,
                  pure_unique_tail_mean=float(np.mean([x['pure_unique_both_directions'] for x in rows])),
                  incremental_tail_mean=float(np.mean([x['incremental_both_directions'] for x in rows])))
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'windows'}))


if __name__ == '__main__':
    main()
