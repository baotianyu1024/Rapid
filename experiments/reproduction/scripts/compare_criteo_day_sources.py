"""Compare day6/day7 source shards using the same saved multihot lookup.

This is a source-distribution diagnostic, not the globally shuffled full-day run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from run_case import LENGTHS


EMBEDDINGS = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
    40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14,
    40000000, 40000000, 40000000, 590152, 12973, 108, 36,
]


def measure(files, lookup_dir, ranks=8, batch=8192, chunks_per_file=3):
    tables = [np.load(lookup_dir/f'table_{t}.npy', mmap_mode='r') for t in range(26)]
    records = []
    columns = [f'categorical_feature_{t+1}' for t in range(26)]
    global_batch = ranks*batch
    for file in files:
        pf = pq.ParquetFile(file)
        full_chunks = pf.metadata.num_rows//global_batch
        if full_chunks < chunks_per_file:
            raise ValueError(f'Not enough complete chunks in {file}')
        chosen = set(np.linspace(0, full_chunks-1, chunks_per_file, dtype=int))
        for chunk_index, chunk in enumerate(pf.iter_batches(batch_size=global_batch,
                                                               columns=columns,
                                                               use_threads=False)):
            if chunk_index not in chosen:
                continue
            if chunk.num_rows != global_batch:
                continue
            sparse = chunk.to_pydict()
            per_table = np.zeros((26, ranks), dtype=np.int64)
            for t, column in enumerate(columns):
                ids = np.fromiter((0 if v is None or v == '' else int(v, 16) % EMBEDDINGS[t]
                                   for v in sparse[column]), dtype=np.int32, count=global_batch)
                expanded = tables[t][ids].reshape(ranks, batch, LENGTHS[t])
                for r in range(ranks):
                    per_table[t, r] = np.unique(expanded[r]).size
            records.append(per_table)
            print(f'{file.name}: chunk {chunk_index}, per-rank unique '
                  f'{per_table.sum(axis=0).mean():.1f}', flush=True)
    counts = np.stack(records)
    return dict(files=[str(p.resolve()) for p in files],
                rows_measured=int(len(records)*global_batch),
                global_batches=len(records), ranks=ranks,
                batch_size_per_rank=batch,
                multihot_lookup_seed=0,
                unique_including_local_per_rank_batch=float(counts.sum(axis=1).mean()),
                per_table_unique_including_local=counts.mean(axis=(0,2)).tolist(),
                batch_means=counts.sum(axis=1).mean(axis=1).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day6-files', nargs='+', type=Path, required=True)
    ap.add_argument('--day7-files', nargs='+', type=Path, required=True)
    ap.add_argument('--lookup-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = {'day6': measure(args.day6_files, args.lookup_dir),
              'day7': measure(args.day7_files, args.lookup_dir)}
    result['day7_minus_day6'] = (result['day7']['unique_including_local_per_rank_batch']-
                                  result['day6']['unique_including_local_per_rank_batch'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: (v if isinstance(v, float) else v['unique_including_local_per_rank_batch'])
                      for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
