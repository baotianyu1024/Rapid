"""Count EMBark remote embedding payload for the compact T-series shards."""

import json
import sys
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'upstream_fresh/HugeCTR/benchmarks/embedding_collection/hugectr'))
from sharding import generate_plan  # noqa: E402


ROOT = Path('/tmp/rapid_tseries_d32_20261104')
for label, case in [('T180', 't180'), ('T200', 't200'), ('T510', 't510_verified')]:
    meta = json.loads((ROOT / case / 'data/input.json').read_text())
    sizes = [table['unique_ids'] for table in meta['tables']]
    args = Namespace(
        optimizer='adagrad', dp_threshold=0, dense_threshold=0,
        sharding_plan='auto', num_gpus_per_node=8,
        memory_cap_for_embedding=20, mem_comm_bw_ratio=2000 / 25,
        mem_comm_work_ratio=8 / 2, dense_comm_work_ratio=4 / 2,
        batchsize=512, use_column_wise_shard=False,
    )
    matrix, strategies, unique, reduction = generate_plan(
        sizes, meta['lengths'], [32] * len(sizes), 1, 8, args, False,
    )
    if (meta['rows'], meta['ranks'], meta['local_batch']) != (65536, 8, 512):
        raise ValueError(f'{label}: unexpected workload')
    if unique or len(reduction) != len(sizes) or [kind for kind, _ in strategies] != ['mp']:
        raise ValueError(f'{label}: planner did not select all Reduction')
    if sorted(int(table) for rank in matrix for table in rank) != list(range(len(sizes))):
        raise ValueError(f'{label}: unexpected table placement')
    steps_per_epoch = meta['rows'] // (8 * 512)
    payload_gib = len(sizes) * 512 * 7 * 8 * steps_per_epoch * 5 * 2 * 32 * 4 / 2**30
    print(f'{label}: Reduction={len(reduction)}, forward+backward={payload_gib:.3f} GiB')
