"""Reassign a complete already-mapped global batch using per-sample owner hits.

Every input batch contains exactly B samples from every rank.  The output is a
capacity-exact permutation of that same batch.  Owners are encoded by ID % G.
This is an offline alternating-optimization step for owner-refined traces.
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
dest = out / "scheduled"
dest.mkdir()
files = [(dest / f"dataset_{r}.bin").open("wb") for r in range(G)]
rng = np.random.default_rng(42)
batch_meta = []
try:
    for b in range(NB):
        records = np.concatenate([x[b * B:(b + 1) * B] for x in arrays])
        n = len(records)
        hits = np.zeros((n, G), dtype=np.int16)
        sparse = records[:, 1 + dense:]
        for t, k in enumerate(lengths):
            values = np.sort(sparse[:, offsets[t]:offsets[t + 1]], axis=1)
            first = np.ones(values.shape, dtype=bool)
            first[:, 1:] = values[:, 1:] != values[:, :-1]
            owners = values % G
            for r in range(G):
                hits[:, r] += np.count_nonzero(first & (owners == r), axis=1).astype(np.int16)
        ranked = np.argsort(-hits, axis=1, kind="stable")
        top = hits[np.arange(n), ranked[:, 0]].astype(np.int32)
        second = hits[np.arange(n), ranked[:, 1]].astype(np.int32)
        jitter = rng.permutation(n)
        order = np.lexsort((jitter, -top, -(top - second)))
        remaining = np.full(G, B, dtype=np.int32)
        buckets = [[] for _ in range(G)]
        for i in order:
            choices = ranked[i]
            available = choices[remaining[choices] > 0]
            if not len(available):
                raise AssertionError("no rank capacity")
            r = int(available[0])
            buckets[r].append(int(i))
            remaining[r] -= 1
        assert not remaining.any() and sum(map(len, buckets)) == n
        assert len(np.unique(np.concatenate([np.asarray(x) for x in buckets]))) == n
        before_hits = sum(int(hits[i, r]) for r in range(G) for i in range(r * B, (r + 1) * B))
        after_hits = sum(int(hits[i, r]) for r in range(G) for i in buckets[r])
        for r in range(G):
            records[np.asarray(buckets[r], dtype=np.int64)].tofile(files[r])
        batch_meta.append({"batch": b, "raw_unique_within_sample_hits_before": before_hits, "raw_unique_within_sample_hits_after": after_hits})
        print("rescheduled batch", b + 1, "/", NB, "hit gain", after_hits - before_hits, flush=True)
finally:
    for f in files:
        f.close()

metrics, actual = measure([dest / f"dataset_{r}.bin" for r in range(G)], G, B, NB, lengths, dense, True)
np.savez_compressed(out / "cps_counts.npz", **actual)
(out / "cps_metrics.json").write_text(json.dumps(metrics, indent=2))
shutil.copyfile(case / "hash_metrics.json", out / "hash_metrics.json")
(out / "reschedule.json").write_text(json.dumps({
    "method": "capacity-exact per-sample max-owner-hit assignment, confidence ordered",
    "base_case": str(case),
    "guarantee": "each global batch is an exact sample permutation; owner and physical ID mapping unchanged",
    "batches": batch_meta,
}, indent=2))
cfg.update(repo=str(Path(__file__).resolve().parents[1] / "working"), base_case=str(case), mapped_trace_reschedule=True)
(out / "config.json").write_text(json.dumps(cfg, indent=2))
(out / "code_sha256.json").write_text(json.dumps({str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
(out / "complete.json").write_text(json.dumps({"completed": time.time()}))
print("mapped trace reschedule complete", out, flush=True)
