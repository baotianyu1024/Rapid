"""Audit an official HugeCTR shard and measure hash-partition ACP payload."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np


DIM_GROUPS = {
    'T180': [128, 64, 64, 32, 128, 128, 256, 128, 128, 64, 128, 64, 64, 128, 32],
    'T200': [128] * 12 + [64] + [128] * 4,
    'T510': [128] * 7,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shard_dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--local-batch', type=int, default=512)
    args = parser.parse_args()
    manifest = json.loads((args.shard_dir / 'manifest.json').read_text())
    tree = ast.parse(Path(manifest['generator']).read_text())
    info = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'dataset_info'
                                                       for t in node.targets))
    rows, ranks = manifest['rows'], args.ranks
    local_batch = args.local_batch
    if local_batch <= 0 or rows % (ranks * local_batch):
        raise ValueError('rows must divide evenly into global batches')
    batches = rows // (ranks * local_batch)
    group_dims = DIM_GROUPS[manifest['dataset']]
    if len(info) != len(group_dims):
        raise ValueError('dimension groups differ from official benchmark')
    table_info = [(hotness, vocab, dim) for (count, hotness, vocab), dim in zip(info, group_dims)
                  for _ in range(count)]
    if len(table_info) != manifest['tables']:
        raise ValueError('table count differs from manifest')
    width = 14 + manifest['sparse_width']
    data = np.memmap(manifest['output'], dtype='<u4', mode='r').reshape(rows, width)
    labels = data[:, 0]
    dense = data[:, 1:14].view('<f4')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(dense).all():
        raise ValueError('invalid labels or dense features')

    reduction_remote = local_batch * (ranks - 1)
    offset = 14
    tables = []
    for table, (hotness, vocab, dim) in enumerate(table_info):
        values = data[:, offset:offset + hotness]
        if int(values.max()) >= vocab:
            raise ValueError(f'ID exceeds vocabulary at table {table}')
        remote_total = 0
        for batch in range(batches):
            batch_values = values[batch * ranks * local_batch:(batch + 1) * ranks * local_batch]
            for rank in range(ranks):
                local = batch_values[rank * local_batch:(rank + 1) * local_batch]
                unique = np.unique(local)
                remote_total += int(np.count_nonzero(unique % ranks != rank))
        unique_remote = remote_total / (ranks * batches)
        strategy = 'Unique' if unique_remote < reduction_remote else 'Reduction'
        tables.append(dict(table=table, hotness=hotness, vocabulary=vocab, dim=dim,
                           unique_remote_per_rank=unique_remote,
                           reduction_remote_per_rank=reduction_remote,
                           strategy=strategy))
        offset += hotness
    if offset != width:
        raise ValueError('table widths do not cover the sparse input')

    selected_bytes = sum((t['unique_remote_per_rank'] if t['strategy'] == 'Unique'
                          else reduction_remote) * t['dim'] * 4 * 2 * ranks * batches for t in tables)
    all_unique_bytes = sum(t['unique_remote_per_rank'] * t['dim'] * 4 * 2 * ranks * batches for t in tables)
    all_reduction_bytes = sum(reduction_remote * t['dim'] * 4 * 2 * ranks * batches for t in tables)
    result = dict(source=manifest['output'], dataset=manifest['dataset'], rows=rows, ranks=ranks,
                  local_batch=local_batch, batches=batches,
                  unique_tables=sum(t['strategy'] == 'Unique' for t in tables),
                  reduction_tables=sum(t['strategy'] == 'Reduction' for t in tables),
                  selected_payload_gib=selected_bytes / 2**30,
                  all_unique_payload_gib=all_unique_bytes / 2**30,
                  all_reduction_payload_gib=all_reduction_bytes / 2**30,
                  note='One shard, hash owner=id%ranks, forward+backward vector payload; not RAPID placement',
                  tables=tables)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'tables'}), flush=True)


if __name__ == '__main__':
    main()
