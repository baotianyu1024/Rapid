# Experimental HugeCTR runtime plan

`cps/generate_local_dataset --mapping_only` emits `embedding_mapping.tsv`
(`table`, original ID, owner rank, physical ID) and one little-endian uint64
`sample_indices_<rank>.bin` per rank. The index files contain absolute source
row numbers in local output order. No dataset shards are written. This export
represents the initial CPS schedule. `experiments/export_runtime_plan.py` can
alternatively reconstruct the final optimized schedule from an existing full
offline run; it verifies complete source records within every global batch.
That offline optimization still materializes temporary transformed shards.

The final export also writes `acp.json` from the final CPS trace profile.
For each table, the default compares mean remote Unique vectors per rank with
`local_batch * (ranks - 1)` remote Reduction vectors. This is an offline
decision, not a per-step runtime profiler.
The separate `experiments/acp/export_hugectr_acp.py` can export from an existing
`cps_metrics.json`; `--cost-model paper` retains the historical comparison of
remote Unique with the paper's total-payload `B*G` expression for reproducing
old results, although these two counts have different scopes.
The HugeCTR training scripts load the static ACP plan before constructing the
embedding collection, validate its layout and apply the two native HugeCTR
compression strategies per table. Counts are in-sample trace estimates, not
measurements of actual GPU link bytes or a guarantee of optimal runtime.

The modified HugeCTR MultiHot RawAsync reader enables the plan only when
`RAPID_PLAN_SOURCE` exactly matches the reader source path. Set `RAPID_PLAN_DIR`
to the plan directory, `RAPID_DENSE_COUNT` to the dense feature count, and
`RAPID_TABLE_LENGTHS` to comma-separated per-table hotness values. The reader
requires a single host, one raw slot, disabled reader shuffle, full global
batches, and a plan covering every requested batch and ID. It applies a
permutation and ID lookup to pinned memory before upload, leaving the input
file unchanged. Every global batch is validated as a permutation.

For repeated datasets that fit in the host page cache, the modified reader
supports `HUGECTR_BUFFERED_IO=1`; the default remains direct I/O. A warm-cache
measurement excludes the first read and must be reported separately from cold
or larger-than-memory workloads. The plan uses bounded direct-index tables (up
to 1 GiB total) with a hash fallback. `RAPID_MAPPING_THREADS` sets the mapping
thread count from 1 to 128 (default 8). Plan loading still validates the TSV
and builds the indices, so its startup time and memory must be measured too.

`experiments/incremental/update_runtime_plan.py` writes a new plan directory,
retaining all existing owners, physical IDs, and sample indices while adding
new complete batches. Its greedy locality heuristic is distinct from the
historical offline RAPID planner. With `RAPID_PLAN_LATEST` set to a marker file,
the reader waits up to 120 seconds for an atomically published extended plan
when it reaches an uncovered batch. `--publish` writes that marker after the
new files are complete. The reader validates that all previously published
mappings and sample indices remain unchanged before switching. The source file
and configured sample count must already cover the future batches, and static
table capacities must accommodate the new physical IDs. SGD live switching was
smoke-tested; checkpoint restoration, multi-host operation, other optimizer
states, and dynamic growth of the input file are not yet validated.
