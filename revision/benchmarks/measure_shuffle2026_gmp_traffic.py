"""Count GMP remote unique embedding requests from its balanced partition."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.case / 'data/input.json').read_text())
    sparse = np.load(args.stage / 'sparse.npy', mmap_mode='r')
    with np.load(args.stage / 'partition_balanced.npz') as partition:
        assignment = partition['data_partition']
        owner = partition['embed_partition']
        mirrors = [partition[str(rank)] for rank in range(8)]
    offsets = meta['active_id_offsets']
    columns = np.cumsum([0] + meta['lengths'])
    counts = np.zeros((16, 8, len(meta['lengths'])), dtype=np.int64)
    for rank in range(8):
        rows = np.flatnonzero(assignment == rank)
        if len(rows) != 16 * 8192:
            raise ValueError(f'Rank {rank} has {len(rows)} rows')
        mirror = np.zeros(len(owner), dtype=np.bool_)
        mirror[mirrors[rank]] = True
        for batch in range(16):
            row = rows[batch*8192:(batch+1)*8192]
            for table in range(len(meta['lengths'])):
                ids = np.unique(sparse[row, columns[table]:columns[table+1]])
                counts[batch, rank, table] = np.count_nonzero(
                    (owner[ids] != rank) & ~mirror[ids])
            print(f'rank {rank} batch {batch+1}/16', flush=True)
    result = dict(unit='remote unique embedding requests per rank per local batch; forward trace estimate',
                  caveat='GMP mirror placement included; gradient and metadata bytes excluded',
                  mean_total_per_rank_batch=float(counts.sum(axis=2).mean()),
                  per_table_mean=counts.mean(axis=(0, 1)).tolist(),
                  counts=counts.tolist())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('mean remote vectors', result['mean_total_per_rank_batch'])


if __name__ == '__main__':
    main()
