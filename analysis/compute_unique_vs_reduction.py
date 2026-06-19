"""
Compute per-table Unique vs Reduction data transfer volume.

Reproduces Figure 4 from the paper:
  (a) Per-table comparison of Unique and Reduction strategies
  (b) Adaptive (ACP) vs fixed strategy total volume

Setup: 8 GPUs, batch_size=8192 per GPU, hash partition.

Usage:
  python scripts/compute_unique_vs_reduction.py \
      --input data/day_7_multihot.bin \
      --num_batches 20 \
      --output scripts/fig4_results
"""
import argparse
import os
import time

import numpy as np


MULTI_HOT_SIZES = [
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1,
    6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1,
]
TOTAL_COLUMNS = 228
SPARSE_START = 14
NUM_TABLES = 26


def build_table_slices():
    slices = []
    offset = 0
    for s in MULTI_HOT_SIZES:
        slices.append((offset, offset + s))
        offset += s
    return slices


TABLE_SLICES = build_table_slices()


def compute_transfer_volumes(input_path, num_gpus, batch_size_per_gpu, num_batches):
    global_batch = num_gpus * batch_size_per_gpu
    num_samples = global_batch * num_batches

    raw = np.fromfile(input_path, dtype=np.int32,
                      count=num_samples * TOTAL_COLUMNS)
    data = raw.reshape(num_samples, TOTAL_COLUMNS)
    sparse = data[:, SPARSE_START:]
    print(f"Loaded {num_samples:,} samples ({num_batches} batches)")

    avg_unique = np.zeros(NUM_TABLES)
    avg_reduction = np.zeros(NUM_TABLES)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * global_batch
        batch_sparse = sparse[batch_start:batch_start + global_batch]

        for t_id, (start, end) in enumerate(TABLE_SLICES):
            table_data = batch_sparse[:, start:end]

            total_unique_remote = 0
            total_reduction_cost = 0

            for g in range(num_gpus):
                gpu_start = g * batch_size_per_gpu
                gpu_end = (g + 1) * batch_size_per_gpu
                gpu_data = table_data[gpu_start:gpu_end]

                unique_ids = np.unique(gpu_data.ravel())
                remote_mask = (unique_ids % num_gpus) != g
                total_unique_remote += remote_mask.sum()

                for s in range(batch_size_per_gpu):
                    target_gpus = set(int(x) % num_gpus for x in gpu_data[s])
                    target_gpus.discard(g)
                    total_reduction_cost += len(target_gpus)

            avg_unique[t_id] += total_unique_remote / num_gpus
            avg_reduction[t_id] += total_reduction_cost / num_gpus

        print(f"  Batch {batch_idx+1}/{num_batches} done")

    avg_unique /= num_batches
    avg_reduction /= num_batches
    return avg_unique, avg_reduction


def generate_figures(avg_unique, avg_reduction, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Select representative tables
    selected = [0, 4, 9, 19, 20, 21]
    selected_names = [f'T{t}\n(pool={MULTI_HOT_SIZES[t]})' for t in selected]

    # Figure (a)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x = np.arange(len(selected))
    width = 0.35

    ax.bar(x - width/2, [avg_unique[t]/1000 for t in selected], width,
           label='Unique', color='#42A5F5', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, [avg_reduction[t]/1000 for t in selected], width,
           label='Reduction', color='#EF5350', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Embedding Table', fontsize=10)
    ax.set_ylabel('Data Transfer Volume (×1000 vectors)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(selected_names, fontsize=8)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    t20_idx = selected.index(20)
    ratio_20 = avg_unique[20] / avg_reduction[20]
    ax.annotate(f'{ratio_20:.1f}×', xy=(t20_idx - width/2, avg_unique[20]/1000),
                xytext=(t20_idx - width/2 - 0.3, avg_unique[20]/1000 + 10),
                fontsize=8, fontweight='bold', color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1))

    plt.tight_layout()
    path_a = os.path.join(output_dir, 'comparison_unique_reduction.pdf')
    plt.savefig(path_a, bbox_inches='tight')
    plt.close()

    # Figure (b)
    total_unique = avg_unique.sum()
    total_reduction = avg_reduction.sum()
    total_adaptive = sum(min(avg_unique[t], avg_reduction[t]) for t in range(NUM_TABLES))

    fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
    strategies = ['All-Reduction', 'All-Unique', 'Adaptive\n(ACP)']
    values = [total_reduction/1000, total_unique/1000, total_adaptive/1000]
    colors = ['#EF5350', '#42A5F5', '#66BB6A']

    bars = ax2.bar(strategies, values, color=colors, edgecolor='white',
                   linewidth=0.8, width=0.5, zorder=3)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.0f}K', ha='center', va='bottom', fontsize=9, fontweight='bold')

    pct = total_adaptive / total_unique * 100
    ax2.text(2, total_adaptive/1000/2, f'{pct:.0f}%\nof Unique',
             ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax2.set_ylabel('Total Data Transfer Volume\n(×1000 vectors)', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--', zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path_b = os.path.join(output_dir, 'ACP_benefit.pdf')
    plt.savefig(path_b, bbox_inches='tight')
    plt.close()

    print(f"Saved: {path_a}")
    print(f"Saved: {path_b}")
    print(f"\nKey numbers:")
    print(f"  Table 20 (pool=100) Unique/Reduction: {ratio_20:.1f}x")
    print(f"  ACP = {pct:.0f}% of best fixed (Unique)")
    print(f"  ACP reduces {(1-total_adaptive/total_reduction)*100:.0f}% vs all-Reduction")


def main():
    parser = argparse.ArgumentParser(description="Compute Unique vs Reduction (Fig 4)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--output", default="scripts/fig4_results")
    args = parser.parse_args()

    start = time.time()
    avg_unique, avg_reduction = compute_transfer_volumes(
        args.input, args.num_gpus, args.batch_size, args.num_batches)

    generate_figures(avg_unique, avg_reduction, args.output)

    np.savez(os.path.join(args.output, 'fig4_data.npz'),
             avg_unique=avg_unique, avg_reduction=avg_reduction)
    print(f"\nDone in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
