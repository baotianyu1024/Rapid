"""Exact best-owner lower bound for a fixed scheduled trace.

For each table/ID, choose the rank that requests it in the largest number of
(batch, destination-rank) cells.  This minimizes total Unique communication
for the fixed sample assignment, but intentionally has no storage constraint;
it is a diagnostic lower bound, not a selectable experiment result.
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--case", required=True)
p.add_argument("--output", required=True)
p.add_argument("--tie-policy", choices=["current", "loaded", "target"], default="current")
p.add_argument("--target-rank", type=int)
p.add_argument("--target-fraction", type=float, default=1.0)
p.add_argument("--target-rank-2", type=int)
p.add_argument("--target-fraction-2", type=float, default=0.0)
p.add_argument("--target-rank-3", type=int)
p.add_argument("--target-fraction-3", type=float, default=0.0)
a = p.parse_args()
case = Path(a.case).resolve()
cfg = json.loads((case / "config.json").read_text())
G, NB, B = cfg["ranks"], cfg["batches"], cfg.get("batch_size", 8192)
lengths, dense = cfg["table_lengths"], cfg["dense"]
width = 1 + dense + sum(lengths)
offsets = np.cumsum([0] + lengths)
arrays = [
    np.memmap(case / "scheduled" / f"dataset_{r}.bin", dtype="<u4", mode="r").reshape(-1, width)
    for r in range(G)
]
base = np.load(case / "cps_counts.npz")
unique = base["unique"]
new_hits = np.zeros_like(unique)
storage = np.zeros(G, dtype=np.int64)
current_storage = np.zeros(G, dtype=np.int64)
table_rows = []

for t in range(len(lengths)):
    lists = []
    pair_parts = []
    col0 = 1 + dense + offsets[t]
    col1 = 1 + dense + offsets[t + 1]
    for b in range(NB):
        for r in range(G):
            ids = np.unique(arrays[r][b * B:(b + 1) * B, col0:col1])
            lists.append(ids)
            pair_parts.append(ids.astype(np.uint64) * np.uint64(G) + np.uint64(r))
    pairs, weights = np.unique(np.concatenate(pair_parts), return_counts=True)
    pair_ids = pairs // np.uint64(G)
    pair_ranks = (pairs % np.uint64(G)).astype(np.int32)
    ids, starts, id_counts = np.unique(pair_ids, return_index=True, return_counts=True)
    # Vectorized segmented reductions keep multi-million-ID tables practical.
    max_weight = np.maximum.reduceat(weights, starts)
    repeated_max = np.repeat(max_weight, id_counts)
    if a.tie_policy == "loaded":
        base_rank_load = base["receive"][:, :, :].sum(axis=1).mean(axis=0)
        priority = np.empty(G, dtype=np.int32)
        priority[np.argsort(-base_rank_load, kind="stable")] = np.arange(G)
        best_priority = np.minimum.reduceat(
            np.where(weights == repeated_max, priority[pair_ranks], G), starts
        )
        rank_by_priority = np.argsort(priority)
        min_best_rank = rank_by_priority[best_priority].astype(np.int32)
    else:
        min_best_rank = np.minimum.reduceat(
            np.where(weights == repeated_max, pair_ranks, G), starts
        ).astype(np.int32)
    current = (ids % np.uint64(G)).astype(np.int32)
    pair_current = (pair_ids % np.uint64(G)).astype(np.int32)
    current_weight = np.maximum.reduceat(
        np.where(pair_ranks == pair_current, weights, 0), starts
    )
    if a.tie_policy == "current":
        owners = np.where(current_weight == max_weight, current, min_best_rank)
    elif a.tie_policy == "target":
        if a.target_rank is None or not 0 <= a.target_rank < G:
            raise ValueError("--tie-policy target requires a valid --target-rank")
        target_is_best = np.logical_or.reduceat(
            (weights == repeated_max) & (pair_ranks == a.target_rank), starts
        )
        if not 0 <= a.target_fraction <= 1:
            raise ValueError("--target-fraction must be in [0,1]")
        selected = (ids % np.uint64(10000)) < int(round(a.target_fraction * 10000))
        target_is_best &= selected
        owners = np.where(target_is_best, a.target_rank,
                          np.where(current_weight == max_weight, current, min_best_rank))
        if a.target_rank_2 is not None:
            if not 0 <= a.target_rank_2 < G or not 0 <= a.target_fraction_2 <= 1:
                raise ValueError("invalid second target rank/fraction")
            target2_is_best = np.logical_or.reduceat(
                (weights == repeated_max) & (pair_ranks == a.target_rank_2), starts
            )
            mod = ids % np.uint64(10000)
            lo = int(round(a.target_fraction * 10000))
            hi = int(round((a.target_fraction + a.target_fraction_2) * 10000))
            owners[target2_is_best & (mod >= lo) & (mod < hi)] = a.target_rank_2
            if a.target_rank_3 is not None:
                if not 0 <= a.target_rank_3 < G or not 0 <= a.target_fraction_3 <= 1:
                    raise ValueError("invalid third target rank/fraction")
                target3_is_best = np.logical_or.reduceat(
                    (weights == repeated_max) & (pair_ranks == a.target_rank_3), starts
                )
                hi3 = int(round((a.target_fraction + a.target_fraction_2 + a.target_fraction_3) * 10000))
                owners[target3_is_best & (mod >= hi) & (mod < hi3)] = a.target_rank_3
    else:
        owners = min_best_rank
    owners = owners.astype(np.int32)
    gains = max_weight.astype(np.int32) - current_weight.astype(np.int32)
    storage += np.bincount(owners, minlength=G)
    current_storage += np.bincount((ids % np.uint64(G)).astype(np.int32), minlength=G)
    for q, values in enumerate(lists):
        pos = np.searchsorted(ids, values)
        new_hits[q // G, t, q % G] = np.count_nonzero(owners[pos] == q % G)
    table_rows.append({
        "table": t,
        "observed_ids": int(len(ids)),
        "moved_ids": int(np.count_nonzero(owners != (ids % np.uint64(G)))),
        "total_local_cell_gain": int(gains.sum()),
    })
    print("owner oracle table", t, "ids", len(ids), "gain", int(gains.sum()), flush=True)

receive = unique - new_hits
per_batch_mean = receive.sum(axis=1).mean(axis=1)
per_batch_peak = receive.sum(axis=1).max(axis=1)
result = {
    "case": str(case),
    "diagnostic_only": True,
    "constraint": "none; exact minimum total Unique communication for this fixed sample assignment",
    "tie_policy": a.tie_policy,
    "ranks": G,
    "batches": NB,
    "receive_mean": float(per_batch_mean.mean()),
    "receive_mean_batch_peak": float(per_batch_peak.mean()),
    "receive_mean_batch_peak_over_mean": float((per_batch_peak / per_batch_mean).mean()),
    "receive_worst_batch_peak_over_mean": float((per_batch_peak / per_batch_mean).max()),
    "per_batch_mean": per_batch_mean.tolist(),
    "per_batch_peak": per_batch_peak.tolist(),
    "receive_matrix": receive.sum(axis=1).tolist(),
    "current_observed_storage": current_storage.tolist(),
    "oracle_observed_storage": storage.tolist(),
    "current_storage_max_over_mean": float(current_storage.max() / current_storage.mean()),
    "oracle_storage_max_over_mean": float(storage.max() / storage.mean()),
    "tables": table_rows,
}
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ["receive_mean", "receive_mean_batch_peak", "receive_mean_batch_peak_over_mean", "receive_worst_batch_peak_over_mean", "oracle_storage_max_over_mean"]}, indent=2))
