"""Independently verify and finalize an already materialized owner refinement."""
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
a = p.parse_args()
case, out = Path(a.case).resolve(), Path(a.output).resolve()
cfg = json.loads((case / "config.json").read_text())
G, NB, B = cfg["ranks"], cfg["batches"], cfg.get("batch_size", 8192)
lengths, dense = cfg["table_lengths"], cfg["dense"]
width = 1 + dense + sum(lengths)
offsets = np.cumsum([0] + lengths)
base_counts = np.load(case / "cps_counts.npz")
storage = np.zeros(G, dtype=np.int64)
inverse_maps = []
table_meta = []
for t in range(len(lengths)):
    m = np.load(out / f"table_{t}_remap.npz")
    old, new, owner = m["old"], m["new"], m["owner"]
    assert len(np.unique(old)) == len(old) == len(np.unique(new))
    assert np.array_equal((new % G).astype(np.int32), owner)
    order = np.argsort(new)
    inverse_maps.append((new[order], old[order]))
    storage += np.bincount(owner, minlength=G)
    table_meta.append({"table": t, "observed_ids": int(len(old)), "moved_owner_ids": int(np.count_nonzero(old % G != owner))})

for r in range(G):
    before = np.memmap(case / "scheduled" / f"dataset_{r}.bin", dtype="<u4", mode="r").reshape(-1, width)
    after = np.memmap(out / "scheduled" / f"dataset_{r}.bin", dtype="<u4", mode="r").reshape(-1, width)
    assert before.shape == after.shape
    for start in range(0, len(before), 32768):
        restored = after[start:start + 32768].copy()
        assert np.array_equal(restored[:, :1 + dense], before[start:start + 32768, :1 + dense])
        for t, (new, old) in enumerate(inverse_maps):
            cols = restored[:, 1 + dense + offsets[t]:1 + dense + offsets[t + 1]]
            pos = np.searchsorted(new, cols)
            safe = np.minimum(pos, len(new) - 1)
            assert not np.any(pos == len(new)) and np.array_equal(new[safe], cols)
            cols[:] = old[safe]
        assert np.array_equal(restored, before[start:start + 32768])
    print("inverse-verified rank", r + 1, "/", G, flush=True)

metrics, actual = measure([out / "scheduled" / f"dataset_{r}.bin" for r in range(G)], G, B, NB, lengths, dense, True)
assert np.array_equal(actual["unique"], base_counts["unique"])
np.savez_compressed(out / "cps_counts.npz", **actual)
(out / "cps_metrics.json").write_text(json.dumps(metrics, indent=2))
shutil.copyfile(case / "hash_metrics.json", out / "hash_metrics.json")
(out / "owner_refinement.json").write_text(json.dumps({
    "method": "exact best owner per observed ID for fixed offline sample assignment",
    "base_case": str(case),
    "verification": "every rank record inverse-mapped exactly to the base record; unique counters preserved",
    "sample_assignment_changed": False,
    "storage_counts": storage.tolist(),
    "storage_max_over_mean": float(storage.max() / storage.mean()),
    "tables": table_meta,
}, indent=2))
cfg.update(repo=str(Path(__file__).resolve().parents[1] / "working"), base_case=str(case), owner_refinement=True, owner_refinement_scope="same offline trace")
(out / "config.json").write_text(json.dumps(cfg, indent=2))
(out / "code_sha256.json").write_text(json.dumps({str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
(out / "complete.json").write_text(json.dumps({"completed": time.time()}))
print("owner refinement inverse-verified and complete", out, flush=True)
