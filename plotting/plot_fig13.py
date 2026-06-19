"""
Plot Figure 13: CPS Effectiveness (Criteo Terabyte).
Colors aligned with the bar charts in Figure 4.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "paper/sigmod_paper/new_images"

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'font.family': 'serif',
})

GPUS = [8, 32, 128]
GPU_LABELS = ['8', '32', '128']
X_POS = np.arange(len(GPUS))
width = 0.35

hash_unique = {8: 466629, 32: 466626, 128: 466658}
cps_unique  = {8: 348334, 32: 286018, 128: 234726}
hash_hit    = {8: 12.5,   32: 3.125,  128: 0.78}
cps_hit     = {8: 69.30,  32: 56.49,  128: 45.15}

# Colors matching bar charts: Hash/Reduction = red, RAPID/Unique-optimized = blue
COLOR_HASH  = '#EF5350'
COLOR_RAPID = '#42A5F5'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))

# (a) Unique remote IDs per device per batch
h_vals = [hash_unique[g] / 1000 for g in GPUS]
c_vals = [cps_unique[g] / 1000 for g in GPUS]

bars1 = ax1.bar(X_POS - width/2, h_vals, width, label='Hash',
                color=COLOR_HASH, edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(X_POS + width/2, c_vals, width, label='RAPID',
                color=COLOR_RAPID, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars1, h_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
             f'{val:.0f}K', ha='center', va='bottom', fontsize=7.5)
for i, (bar, val) in enumerate(zip(bars2, c_vals)):
    pct = (1 - c_vals[i] / h_vals[i]) * 100
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
             f'{val:.0f}K\n({pct:.1f}%↓)', ha='center', va='bottom', fontsize=6.5)

ax1.set_xlabel('Number of GPUs')
ax1.set_ylabel('Unique IDs per Device (×1000)')
ax1.set_xticks(X_POS)
ax1.set_xticklabels(GPU_LABELS)
ax1.legend(fontsize=8)
ax1.set_title('(a) Unique Remote IDs per Device')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# (b) Local hit rate
h_hit = [hash_hit[g] for g in GPUS]
c_hit = [cps_hit[g] for g in GPUS]

bars1 = ax2.bar(X_POS - width/2, h_hit, width, label='Hash',
                color=COLOR_HASH, edgecolor='white', linewidth=0.5)
bars2 = ax2.bar(X_POS + width/2, c_hit, width, label='RAPID',
                color=COLOR_RAPID, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars1, h_hit):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5)
for bar, val in zip(bars2, c_hit):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5)

ax2.set_xlabel('Number of GPUs')
ax2.set_ylabel('Local Hit Rate (%)')
ax2.set_xticks(X_POS)
ax2.set_xticklabels(GPU_LABELS)
ax2.legend(fontsize=8)
ax2.set_title('(b) Local Hit Rate')
ax2.set_ylim(0, 82)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
path = os.path.join(OUT_DIR, 'cps_effectiveness.pdf')
plt.savefig(path, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")
