"""
Ablation Study: 3x3 matrix of Step1 (ID allocation) x Step2 (Sample assignment)

Step 1 (ID allocation):
  - Hash: hash(encode(tid, fid)) % num_devices
  - Freq RR: sort by frequency descending, round-robin
  - Ours: arrange_ids_to_devices_v3

Step 2 (Sample assignment):
  - Sequential: samples assigned by file order (device = sample_pos // batch_size)
  - Simple Greedy: each sample goes to device with most local IDs (balanced)
  - Ours: generate_local_dataset_v2 (gamma=0.2)

Evaluation uses the C++ evaluators for speed.
"""
import numpy as np
import os
import sys
import time
import json
import struct
import hashlib
import subprocess
import argparse
from collections import Counter, defaultdict
from pathlib import Path

# === Parse CLI ===
_parser = argparse.ArgumentParser(description="Ablation: 3x3 Step1 x Step2")
_parser.add_argument("--num_devices", type=int, default=8)
_parser.add_argument("--eval_batches", type=int, default=None, help="Auto = max batches from file")
_parser.add_argument("--input_bin", type=str, default=None)
_parser.add_argument("--hot_ids_dir", type=str, default=None)
_parser.add_argument("--output_tag", type=str, default=None, help="Sub-dir tag for results")
_cli_args = _parser.parse_args()

# === Configuration ===
NUM_DEVICES = _cli_args.num_devices
BATCH_SIZE = 8192
DENSE_COUNT = 13
TABLE_LENGTHS = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
TOTAL_SPARSE = sum(TABLE_LENGTHS)
RECORD_SIZE = 4 + DENSE_COUNT * 4 + TOTAL_SPARSE * 4
GLOBAL_BATCH = NUM_DEVICES * BATCH_SIZE
SEED = 42

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROJECT_ROOT = BASE_DIR.parent

if _cli_args.input_bin:
    INPUT_BIN = Path(_cli_args.input_bin)
else:
    INPUT_BIN = DATA_DIR / "input.bin"

if _cli_args.hot_ids_dir:
    HOT_IDS_DIR = Path(_cli_args.hot_ids_dir)
else:
    HOT_IDS_DIR = DATA_DIR / "hot_ids"

# Auto-compute eval_batches from file size
if _cli_args.eval_batches:
    EVAL_BATCHES = _cli_args.eval_batches
else:
    _file_size = os.path.getsize(INPUT_BIN)
    EVAL_BATCHES = _file_size // (RECORD_SIZE * GLOBAL_BATCH)

_tag = _cli_args.output_tag or f"{NUM_DEVICES}p"
RESULTS_DIR = BASE_DIR / "results" / f"ablation_{_tag}"
TL_STR = " ".join(str(x) for x in TABLE_LENGTHS)


def encode_id(table_id, feature_id):
    return (table_id << 32) | (feature_id & 0xFFFFFFFF)


def get_table_offsets():
    offsets = []
    off = 0
    for tl in TABLE_LENGTHS:
        offsets.append((off, off + tl))
        off += tl
    return offsets


def write_assignment_dir(assignment, output_dir):
    """Write assignment dict to table_0 ... table_N files (vectorized)."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy arrays for fast processing
    eids = np.array(list(assignment.keys()), dtype=np.uint64)
    devs = np.array(list(assignment.values()), dtype=np.int32)
    tids = (eids >> 32).astype(np.int32)
    fids = (eids & 0xFFFFFFFF).astype(np.uint32)

    for d in range(NUM_DEVICES):
        mask = devs == d
        d_tids = tids[mask]
        d_fids = fids[mask]
        with open(os.path.join(output_dir, f"table_{d}"), 'w', buffering=1<<20) as f:
            for t, fid in zip(d_tids.tolist(), d_fids.tolist()):
                f.write(f"{t}:{fid}\n")
    print(f"    Written {len(eids)} IDs to {output_dir}")


def collect_all_ids_and_freq():
    """Collect ALL unique IDs and frequencies using per-table numpy arrays (fast)."""
    print("Collecting IDs and frequencies...")
    dt = np.dtype([
        ('label', 'i4'),
        ('dense', 'f4', (DENSE_COUNT,)),
        ('sparse', 'i4', (TOTAL_SPARSE,))
    ])
    offsets = get_table_offsets()

    # Read all needed rows at once
    total_rows = EVAL_BATCHES * GLOBAL_BATCH
    t0 = time.time()

    data = np.fromfile(str(INPUT_BIN), dtype=dt, count=total_rows)
    actual_rows = len(data)
    print(f"  Read {actual_rows} rows in {time.time()-t0:.1f}s")

    sparse = data['sparse']  # (N, TOTAL_SPARSE)

    # Per-table frequency counting using numpy
    freq = {}  # encode_id -> count
    for tid, (start, end) in enumerate(offsets):
        cols = sparse[:, start:end].ravel()
        vals, counts = np.unique(cols, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            freq[encode_id(tid, v)] = freq.get(encode_id(tid, v), 0) + c
        if (tid + 1) % 5 == 0:
            print(f"  Table {tid+1}/{len(offsets)}, total unique: {len(freq)}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. Total unique IDs: {len(freq)}")
    return freq


# ============================================================
# Step 1: ID Allocation Strategies
# ============================================================

def step1_hash(all_ids, output_dir):
    """Hash-based allocation using fast numeric hash (vectorized)."""
    print("  [Step1] Hash allocation...")
    t0 = time.time()
    eids = np.array(list(all_ids), dtype=np.uint64)
    # Fast integer hash: multiply by large prime, xor with shift
    h = (eids * np.uint64(0x517cc1b727220a95)) ^ (eids >> np.uint64(17))
    devs = (h % np.uint64(NUM_DEVICES)).astype(np.int32)
    assignment = dict(zip(eids.tolist(), devs.tolist()))
    write_assignment_dir(assignment, output_dir)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s, {len(assignment)} IDs")
    return elapsed


def step1_freq_rr(freq, output_dir):
    """Frequency-based round-robin (vectorized sort)."""
    print("  [Step1] Frequency Round-Robin allocation...")
    t0 = time.time()
    eids = np.array(list(freq.keys()), dtype=np.uint64)
    counts = np.array([freq[k] for k in eids.tolist()], dtype=np.int64)
    # Sort by count descending
    order = np.argsort(-counts)
    eids_sorted = eids[order]
    # Round-robin assignment
    devs = np.arange(len(eids_sorted), dtype=np.int32) % NUM_DEVICES
    assignment = dict(zip(eids_sorted.tolist(), devs.tolist()))
    write_assignment_dir(assignment, output_dir)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s, {len(assignment)} IDs")
    return elapsed


def step1_ours(output_dir):
    """Our workload-aware allocation (arrange_ids_to_devices_v3)."""
    print("  [Step1] Ours (arrange_ids_to_devices_v3)...")
    binary = PROJECT_ROOT / "cps" / "arrange_ids_to_devices"
    t0 = time.time()
    cmd = (
        f"OMP_NUM_THREADS=16 {binary} "
        f"--input_bin {INPUT_BIN} "
        f"--hot_ids_dir {HOT_IDS_DIR} "
        f"--output_dir {output_dir} "
        f"--batch_size {BATCH_SIZE} "
        f"--num_devices {NUM_DEVICES} "
        f"--dense_count {DENSE_COUNT} "
        f"--n_batches {EVAL_BATCHES} "
        f"--table_lengths {TL_STR}"
    )
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
    elapsed = time.time() - t0
    # Print last few lines
    lines = result.stdout.strip().split('\n')
    for line in lines[-5:]:
        print(f"    {line}")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
    # Count IDs
    total = 0
    for d in range(NUM_DEVICES):
        fpath = os.path.join(output_dir, f"table_{d}")
        if os.path.exists(fpath):
            with open(fpath) as f:
                total += sum(1 for _ in f)
    print(f"  Done in {elapsed:.1f}s, {total} IDs")
    return elapsed


# ============================================================
# Step 2: Sample Assignment + Evaluation
# ============================================================

def step2_sequential(assignment_dir):
    """
    Sequential: device = sample_position_in_batch // batch_size.
    Uses evaluate_partition (no sample optimization).
    """
    evaluator = BASE_DIR / "evaluate_partition"
    cmd = (
        f"OMP_NUM_THREADS=16 {evaluator} "
        f"--input {INPUT_BIN} "
        f"--assignment_dir {assignment_dir} "
        f"--num_devices {NUM_DEVICES} "
        f"--batch_size {BATCH_SIZE} "
        f"--num_batches {EVAL_BATCHES} "
        f"--dense_count {DENSE_COUNT} "
        f"--table_lengths {TL_STR}"
    )
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
    return parse_eval_output(result.stdout)


def step2_greedy(assignment_dir):
    """
    Simple Greedy: each sample goes to device with most local IDs (balanced).
    Uses evaluate_with_sample_assignment.
    """
    evaluator = BASE_DIR / "evaluate_with_sample_assignment"
    cmd = (
        f"OMP_NUM_THREADS=16 {evaluator} "
        f"--input {INPUT_BIN} "
        f"--assignment_dir {assignment_dir} "
        f"--num_devices {NUM_DEVICES} "
        f"--batch_size {BATCH_SIZE} "
        f"--num_batches {EVAL_BATCHES} "
        f"--dense_count {DENSE_COUNT} "
        f"--table_lengths {TL_STR}"
    )
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
    return parse_eval_output(result.stdout)


def step2_ours(assignment_dir, dataset_output_dir):
    """
    Ours: generate_local_dataset_v2 + verify_hit_rate_from_transformed.
    """
    binary_gen = PROJECT_ROOT / "cps" / "generate_local_dataset"
    binary_verify = PROJECT_ROOT / "analysis" / "verify_hit_rate"

    # Generate dataset
    if os.path.exists(dataset_output_dir):
        raise FileExistsError(dataset_output_dir)
    cmd_gen = (
        f"OMP_NUM_THREADS=8 {binary_gen} "
        f"--input_file {INPUT_BIN} "
        f"--table_files_dir {assignment_dir} "
        f"--output_dir {dataset_output_dir} "
        f"--num_devices {NUM_DEVICES} "
        f"--train_batch_size {BATCH_SIZE} "
        f"--steps_per_chunk 1 "
        f"--num_chunks {EVAL_BATCHES} "
        f"--num_threads 1 "
        f"--dense_count {DENSE_COUNT} "
        f"--gamma 0.2 "
        f"--table_lengths {TL_STR}"
    )
    result = subprocess.run(cmd_gen, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
    if result.returncode != 0:
        print(f"  generate_local_dataset_v2 FAILED: {result.stderr[-500:]}")
        return None

    # Verify
    cmd_verify = (
        f"OMP_NUM_THREADS=8 {binary_verify} "
        f"--no_shuffle --dataset_dir {dataset_output_dir} "
        f"--num_devices {NUM_DEVICES} "
        f"--batch_size {BATCH_SIZE} "
        f"--num_batches {EVAL_BATCHES} "
        f"--dense_count {DENSE_COUNT} "
        f"--table_lengths {TL_STR}"
    )
    result = subprocess.run(cmd_verify, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
    return parse_verify_output(result.stdout)


def parse_eval_output(output):
    """Parse output from evaluate_partition or evaluate_with_sample_assignment."""
    metrics = {}
    for line in output.split('\n'):
        if line.startswith('Locality:'):
            metrics['locality'] = float(line.split(':')[1].strip().rstrip('%'))
        elif line.startswith('Balance factor:') or line.startswith('ID Balance:'):
            metrics['balance_factor'] = float(line.split(':')[1].strip())
        elif line.startswith('Total IDs:'):
            metrics['total_ids'] = int(line.split(':')[1].strip())
        elif line.startswith('Eval time:'):
            metrics['eval_time_s'] = float(line.split(':')[1].strip().rstrip('s'))
    return metrics if metrics else None


def parse_verify_output(output):
    """Parse output from verify_hit_rate_from_transformed."""
    metrics = {}
    for line in output.split('\n'):
        if 'Avg local hit rate/batch' in line:
            val = line.split(':')[1].strip().rstrip('%')
            metrics['locality'] = float(val)
        elif 'Local hits' in line:
            # e.g. "  Local hits               : 20010890 (77.45%)"
            import re
            m = re.search(r'\((\d+\.\d+)%\)', line)
            if m:
                metrics['locality'] = float(m.group(1))
    return metrics if metrics else None


def count_ids_in_dir(assignment_dir):
    """Count total IDs in assignment directory."""
    total = 0
    for d in range(NUM_DEVICES):
        fpath = os.path.join(assignment_dir, f"table_{d}")
        if os.path.exists(fpath):
            with open(fpath) as f:
                total += sum(1 for _ in f)
    return total


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== Ablation Config ===")
    print(f"  NUM_DEVICES: {NUM_DEVICES}")
    print(f"  EVAL_BATCHES: {EVAL_BATCHES}")
    print(f"  GLOBAL_BATCH: {GLOBAL_BATCH}")
    print(f"  INPUT_BIN: {INPUT_BIN}")
    print(f"  HOT_IDS_DIR: {HOT_IDS_DIR}")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")
    print()

    # Compile both evaluators
    print("Compiling evaluators...")
    for src in ["evaluate_partition.cpp", "evaluate_with_sample_assignment.cpp"]:
        name = src.replace('.cpp', '')
        cmd = f"g++ -O3 -std=c++17 -fopenmp -pthread {BASE_DIR}/{src} -o {BASE_DIR}/{name}"
        r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        if r.returncode != 0:
            print(f"  FAILED to compile {src}: {r.stderr}")
            return
    print("  Both evaluators compiled.\n")

    # Collect frequencies
    freq = collect_all_ids_and_freq()
    all_ids = set(freq.keys())
    EXPECTED_TOTAL = len(all_ids)
    print(f"\nExpected total IDs for all strategies: {EXPECTED_TOTAL}\n")

    # ============================================================
    # Step 1: Generate all ID allocations
    # ============================================================
    print("=" * 60)
    print("STEP 1: GENERATING ID ALLOCATIONS")
    print("=" * 60)

    step1_times = {}

    # Hash
    hash_dir = str(RESULTS_DIR / "step1_hash")
    step1_times['Hash'] = step1_hash(all_ids, hash_dir)
    hash_count = count_ids_in_dir(hash_dir)
    print(f"  Verify: {hash_count} IDs (expected {EXPECTED_TOTAL})\n")

    # Freq RR
    freq_rr_dir = str(RESULTS_DIR / "step1_freq_rr")
    step1_times['Freq_RR'] = step1_freq_rr(freq, freq_rr_dir)
    freq_rr_count = count_ids_in_dir(freq_rr_dir)
    print(f"  Verify: {freq_rr_count} IDs (expected {EXPECTED_TOTAL})\n")

    # Ours
    ours_dir = str(RESULTS_DIR / "step1_ours")
    if os.path.exists(ours_dir):
        import shutil
        shutil.rmtree(ours_dir)
    step1_times['Ours'] = step1_ours(ours_dir)
    ours_count = count_ids_in_dir(ours_dir)
    print(f"  Verify: {ours_count} IDs (expected {EXPECTED_TOTAL})\n")

    # Check consistency
    print(f"ID counts: Hash={hash_count}, Freq_RR={freq_rr_count}, Ours={ours_count}")
    if hash_count != freq_rr_count:
        print("  WARNING: Hash and Freq_RR have different ID counts!")
    if ours_count != hash_count:
        print(f"  NOTE: Ours has {ours_count} vs others {hash_count} "
              f"(diff={ours_count - hash_count})")
    print()

    # ============================================================
    # Step 2: Evaluate all 9 combinations
    # ============================================================
    print("=" * 60)
    print("STEP 2: EVALUATING 3x3 MATRIX")
    print("=" * 60)

    step1_dirs = {
        'Hash': hash_dir,
        'Freq_RR': freq_rr_dir,
        'Ours': ours_dir,
    }

    results = {}

    for s1_name, s1_dir in step1_dirs.items():
        for s2_name in ['Sequential', 'Greedy', 'Ours']:
            key = f"{s1_name} + {s2_name}"
            print(f"\n--- {key} ---")

            if s2_name == 'Sequential':
                metrics = step2_sequential(s1_dir)
            elif s2_name == 'Greedy':
                metrics = step2_greedy(s1_dir)
            else:  # Ours
                dataset_dir = str(RESULTS_DIR / f"dataset_{s1_name.lower()}")
                if os.path.exists(dataset_dir):
                    import shutil
                    shutil.rmtree(dataset_dir)
                metrics = step2_ours(s1_dir, dataset_dir)

            if metrics:
                metrics['step1_time_s'] = step1_times[s1_name]
                results[key] = metrics
                print(f"  Locality: {metrics.get('locality', 'N/A')}%")
            else:
                print(f"  FAILED")
            print()

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("ABLATION RESULTS: Step1 (ID Allocation) x Step2 (Sample Assignment)")
    print("=" * 70)

    # Print as matrix
    s2_names = ['Sequential', 'Greedy', 'Ours']
    s1_names = ['Hash', 'Freq_RR', 'Ours']

    step1_label = "Step1 \\ Step2"
    header = f"{step1_label:<15}"
    for s2 in s2_names:
        header += f" {s2:<12}"
    print(header)
    print("-" * 55)

    for s1 in s1_names:
        row = f"{s1:<15}"
        for s2 in s2_names:
            key = f"{s1} + {s2}"
            if key in results:
                loc = results[key].get('locality', 0)
                row += f" {loc:<12.2f}"
            else:
                row += f" {'FAIL':<12}"
        print(row)
    print("=" * 55)

    # Also print preprocessing times
    print(f"\nStep 1 Preprocessing Times:")
    for name, t in step1_times.items():
        print(f"  {name}: {t:.1f}s")

    # Save
    results_file = RESULTS_DIR / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
