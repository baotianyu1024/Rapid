"""Materialize the exact best-ID-owner assignment for a fixed trace.

This is an offline, in-sample alternating-optimization step: sample-to-rank
assignment stays fixed and every observed table/ID is assigned to the rank
that requests it in the most batch/rank cells.  The old-to-new physical-ID
maps make the transformation reversible.  No samples or IDs are dropped.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from measure import measure

p = argparse.ArgumentParser()
p.add_argument("--case", required=True)
p.add_argument("--output", required=True)
p.add_argument("--target-rank", type=int)
p.add_argument("--target-fraction", type=float, default=0.0)
p.add_argument("--target-rank-2", type=int)
p.add_argument("--target-fraction-2", type=float, default=0.0)
p.add_argument("--target-rank-3", type=int)
p.add_argument("--target-fraction-3", type=float, default=0.0)
p.add_argument("--no-base-counts", action="store_true", help="Derive validation counts during owner profiling; no saved trace counts required")
p.add_argument("--uncompressed-maps", action="store_true", help="Store identical mapping arrays without archive compression")
a = p.parse_args()
case, out = Path(a.case).resolve(), Path(a.output).resolve()
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)
cfg = json.loads((case / "config.json").read_text())
G, NB, B = cfg["ranks"], cfg["batches"], cfg.get("batch_size", 8192)
lengths, dense = cfg["table_lengths"], cfg["dense"]
width = 1 + dense + sum(lengths)
offsets = np.cumsum([0] + lengths)
arrays = [np.memmap(case / "scheduled" / f"dataset_{r}.bin", dtype="<u4", mode="r").reshape(-1, width) for r in range(G)]
base = None if a.no_base_counts else np.load(case / "cps_counts.npz")
predicted_hits = np.zeros((NB,len(lengths),G),dtype=np.int64)
predicted_unique = np.zeros_like(predicted_hits)
maps = []
storage = np.zeros(G, dtype=np.int64)
table_meta = []
owner_start = time.perf_counter()

for t in range(len(lengths)):
    lists, pair_parts = [], []
    c0, c1 = 1 + dense + offsets[t], 1 + dense + offsets[t + 1]
    for b in range(NB):
        for r in range(G):
            ids = np.unique(arrays[r][b * B:(b + 1) * B, c0:c1])
            lists.append(ids)
            pair_parts.append(ids.astype(np.uint64) * np.uint64(G) + np.uint64(r))
    pairs, weights = np.unique(np.concatenate(pair_parts), return_counts=True)
    pair_ids = pairs // np.uint64(G)
    pair_ranks = (pairs % np.uint64(G)).astype(np.int32)
    ids, starts, counts = np.unique(pair_ids, return_index=True, return_counts=True)
    max_weight = np.maximum.reduceat(weights, starts)
    repeated_max = np.repeat(max_weight, counts)
    best = np.minimum.reduceat(np.where(weights == repeated_max, pair_ranks, G), starts).astype(np.int32)
    current = (ids % np.uint64(G)).astype(np.int32)
    current_weight = np.maximum.reduceat(
        np.where(pair_ranks == (pair_ids % np.uint64(G)).astype(np.int32), weights, 0), starts
    )
    owners = np.where(current_weight == max_weight, current, best).astype(np.int32)
    if a.target_rank is not None:
        if not 0 <= a.target_rank < G or not 0 <= a.target_fraction <= 1:
            raise ValueError("invalid target rank/fraction")
        target_is_best = np.logical_or.reduceat(
            (weights == repeated_max) & (pair_ranks == a.target_rank), starts
        )
        selected = (ids % np.uint64(10000)) < int(round(a.target_fraction * 10000))
        owners[target_is_best & selected] = a.target_rank
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
    newids = np.empty(len(ids), dtype=np.uint32)
    for r in range(G):
        idx = np.flatnonzero(owners == r)
        values = np.arange(len(idx), dtype=np.uint64) * np.uint64(G) + np.uint64(r)
        if len(values) and values[-1] > np.iinfo(np.uint32).max:
            raise OverflowError("refined physical ID exceeds uint32")
        newids[idx] = values.astype(np.uint32)
    storage += np.bincount(owners, minlength=G)
    for q, values in enumerate(lists):
        pos = np.searchsorted(ids, values)
        predicted_hits[q // G, t, q % G] = np.count_nonzero(owners[pos] == q % G)
        predicted_unique[q // G, t, q % G] = len(values)
    writer = np.savez if a.uncompressed_maps else np.savez_compressed
    writer(out / f"table_{t}_remap.npz", old=ids.astype(np.uint32), new=newids, owner=owners)
    # Exact dense lookup for compact physical IDs. Keep the sparse-range
    # fallback so a very large maximum ID cannot cause an unbounded allocation.
    lut = None
    if len(ids) and int(ids[-1]) + 1 <= max(1000000, 8 * len(ids)):
        lut = np.full(int(ids[-1]) + 1, -1, dtype=np.int64)
        lut[ids] = newids
    maps.append((ids.astype(np.uint32), newids, lut))
    table_meta.append({"table": t, "observed_ids": int(len(ids)), "moved_owner_ids": int(np.count_nonzero(owners != current))})
    print("refined owner table", t, "ids", len(ids), flush=True)

owner_compute_seconds = time.perf_counter() - owner_start
materialize_start = time.perf_counter()
dest = out / "scheduled"
dest.mkdir()
for r in range(G):
    with (dest / f"dataset_{r}.bin").open("wb") as f:
        for start in range(0, NB * B, 32768):
            records = arrays[r][start:min(start + 32768, NB * B)].copy()
            for t, (old, new, lut) in enumerate(maps):
                cols = records[:, 1 + dense + offsets[t]:1 + dense + offsets[t + 1]]
                if lut is not None:
                    mapped = lut[cols]
                    if np.any(mapped < 0):
                        raise AssertionError("scheduled ID missing from reversible remap")
                    cols[:] = mapped
                else:
                    pos = np.searchsorted(old, cols)
                    safe = np.minimum(pos, len(old) - 1)
                    if np.any(pos == len(old)) or np.any(old[safe] != cols):
                        raise AssertionError("scheduled ID missing from reversible remap")
                    cols[:] = new[safe]
            records.tofile(f)
    print("rewrote rank", r + 1, "/", G, flush=True)

materialize_seconds = time.perf_counter() - materialize_start
profile_start = time.perf_counter()
metrics, actual = measure([dest / f"dataset_{r}.bin" for r in range(G)], G, B, NB, lengths, dense, True)
profile_seconds = time.perf_counter() - profile_start
assert np.array_equal(actual["unique"], predicted_unique)
if base is not None: assert np.array_equal(actual["unique"], base["unique"])
assert np.array_equal(actual["receive"], actual["unique"] - predicted_hits)
np.savez_compressed(out / "cps_counts.npz", **actual)
(out / "cps_metrics.json").write_text(json.dumps(metrics, indent=2))
if (case / "hash_metrics.json").exists(): shutil.copyfile(case / "hash_metrics.json", out / "hash_metrics.json")
metadata = {
    "method": "exact best owner per observed ID for fixed offline sample assignment",
    "base_case": str(case),
    "reversible_maps": "table_N_remap.npz old/new physical IDs",
    "sample_assignment_changed": False,
    "tie_target_rank": a.target_rank,
    "tie_target_fraction": a.target_fraction,
    "tie_target_rank_2": a.target_rank_2,
    "tie_target_fraction_2": a.target_fraction_2,
    "tie_target_rank_3": a.target_rank_3,
    "tie_target_fraction_3": a.target_fraction_3,
    "storage_counts": storage.tolist(),
    "storage_max_over_mean": float(storage.max() / storage.mean()),
    "tables": table_meta,
}
(out / "owner_refinement.json").write_text(json.dumps(metadata, indent=2))
cfg.update(repo=str(Path(__file__).resolve().parents[1] / "working"), base_case=str(case), owner_refinement=True, owner_refinement_scope="same offline trace")
(out / "config.json").write_text(json.dumps(cfg, indent=2))
(out / "code_sha256.json").write_text(json.dumps({str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
(out / "complete.json").write_text(json.dumps({"completed": time.time()}))
(out / "phase_times.json").write_text(json.dumps(dict(owner_compute_and_map_write_seconds=owner_compute_seconds,
    materialize_seconds=materialize_seconds,acp_profile_and_count_seconds=profile_seconds),indent=2))
print("owner refinement independently verified", out, flush=True)
