# RAPID: Workload-Aware Partitioning and Cost-Based Planning for Distributed Embedding Tables

## Quick Start

```bash
# 1. Compile C++ tools
bash compile.sh

# 2. Run full pipeline (example: Criteo Terabyte, 8 GPUs)
bash run_all.sh /path/to/original.bin 8 ./output_8p \
    --table_lengths 3 2 1 2 6 1 1 1 1 7 3 8 1 6 9 5 1 1 1 12 100 27 10 3 1 1 \
    --dense_count 13 --topk 40000 --batch_size 8192
```

## Modules

### data_preparation/
- `criteo_singlehot_to_multihot.py` — Convert Criteo TSV (single-hot) to MLPerf multi-hot binary format with shuffle
- `taobao_preprocess.py` — Preprocess Taobao Display Ad CSV files into binary format
- `taobao_extract_4tables.py` — Extract 4 large embedding tables from the 15-sparse Taobao dataset

### cps/ (Co-occurrence-aware Partitioning and Scheduling)
- `hot_id_extraction.py` — Extract top-K hot embedding IDs by frequency
- `arrange_ids_to_devices.cpp` — Streaming greedy ID allocation (Steps 1-4 in the paper)
- `generate_local_dataset.cpp` — Capacity-aware sample scheduling + write mapped dataset (Step 5)
- `merge_for_hugectr.py` — Merge per-device datasets into a single interleaved binary for HugeCTR

### acp/ (Adaptive Data Access Planning)
- `compute_acp.py` — Per-table strategy selection (Unique vs Reduction) based on cost model

### analysis/
- `verify_hit_rate.cpp` — Verify local hit rate of CPS-transformed dataset
- `compute_unique_vs_reduction.py` — Per-table Unique vs Reduction communication volume comparison

### evaluation/
- `run_ablation.py` — 3x3 ablation study (placement x scheduling)
- `evaluate_partition.cpp` — Evaluate partition quality with sequential sample assignment
- `evaluate_with_sample_assignment.cpp` — Evaluate with greedy balanced sample assignment

### training/
- `run_hugectr_dlrm.py` — HugeCTR DLRM training script with multiple dataset modes

### plotting/
- `plot_fig10_fig11.py` — Generate Figure 10 (deduplication rate) and Figure 11 (communication volume scaling)
- `plot_fig13.py` — Generate Figure 13 (CPS effectiveness)

## Binary Dataset Format

All binary datasets use the same record format:
```
label(int32) + dense(float32 × D) + sparse(int32 × S) = record_size bytes per sample
```

| Dataset | Dense (D) | Sparse (S) | Record Size | Table Lengths |
|---------|-----------|------------|-------------|---------------|
| Criteo Terabyte | 13 | 214 | 912 bytes | 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 |
| Criteo Kaggle | 13 | 214 | 912 bytes | (same as above) |
| Taobao (4 tables) | 1 | 4 | 24 bytes | 1,1,1,1 |
