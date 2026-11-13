"""
Plot Figure 10 (deduplication rate) and Figure 11 (communication volume).
Style matches the paper's existing figures.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "paper/sigmod_paper/new_images"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.6,
    'lines.markersize': 5,
    'figure.dpi': 300,
    'font.family': 'serif',
})

GPUS = [8, 16, 32, 64, 128]
GPU_LABELS = ['8p', '16p', '32p', '64p', '128p']
X_POS = np.arange(len(GPUS))  # evenly spaced positions
B = 8192

# ============================================================
# Data (remote embedding vectors per GPU per batch, forward only)
# ============================================================

criteo = {
    8:   {'red': B*7*26,   'uniq': 408443,  'acp': 251367,  'cps': 106854,  'uniq_cps': 106854},
    16:  {'red': B*15*26,  'uniq': 437273,  'acp': 336838,  'cps': 98009,   'uniq_cps': 98009},
    32:  {'red': B*31*26,  'uniq': 452096,  'acp': 452096,  'cps': 124572,  'uniq_cps': 124572},
    64:  {'red': B*63*26,  'uniq': 459324,  'acp': 459324,  'cps': 127033,  'uniq_cps': 127033},
    128: {'red': B*127*26, 'uniq': 463014,  'acp': 463014,  'cps': 128734,  'uniq_cps': 128734},
}

kaggle = {
    8:   {'red': B*7*26,   'uniq': 472324,  'acp': 144482,  'cps': 123308,  'uniq_cps': 178092},
    16:  {'red': B*15*26,  'uniq': 505950,  'acp': 215663,  'cps': 192497,  'uniq_cps': 221698},
    32:  {'red': B*31*26,  'uniq': 522640,  'acp': 349542,  'cps': 247099,  'uniq_cps': 247099},
    64:  {'red': B*63*26,  'uniq': 531182,  'acp': 531182,  'cps': 259544,  'uniq_cps': 259544},
    128: {'red': B*127*26, 'uniq': 535408,  'acp': 535408,  'cps': 267408,  'uniq_cps': 267408},
}

taobao = {
    8:   {'red': B*7*4,   'uniq': 24540,  'acp': 24540,  'cps': 10358,  'uniq_cps': 10358},
    16:  {'red': B*15*4,  'uniq': 26291,  'acp': 26291,  'cps': 11200,  'uniq_cps': 11200},
    32:  {'red': B*31*4,  'uniq': 27167,  'acp': 27167,  'cps': 11314,  'uniq_cps': 11314},
    64:  {'red': B*63*4,  'uniq': 27604,  'acp': 27604,  'cps': 11132,  'uniq_cps': 11132},
    128: {'red': B*127*4, 'uniq': 27823,  'acp': 27823,  'cps': 11085,  'uniq_cps': 11085},
}

SUM_L = {'criteo': 214, 'kaggle': 214, 'taobao': 4}

# Colors and markers matching the original figure style
STYLE = {
    'uniq':     {'color': '#E57373', 'marker': '*',  'ls': '--', 'label': 'Unique', 'fillstyle': 'none'},
    'uniq_cps': {'color': '#A1887F', 'marker': 's',  'ls': '--', 'label': 'Unique+CPS', 'fillstyle': 'none'},
    'acp':      {'color': '#5C6BC0', 'marker': 'o',  'ls': '-',  'label': 'ACP', 'fillstyle': 'full'},
    'cps':      {'color': '#66BB6A', 'marker': '^',  'ls': '-',  'label': 'RAPID (ACP+CPS)', 'fillstyle': 'full'},
    'red':      {'color': '#EF5350', 'marker': 'o',  'ls': '-',  'label': 'All-Reduction', 'fillstyle': 'full'},
}

# ============================================================
# Figure 10: Deduplication Rate
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))

datasets_fig10 = [
    ('Terabyte', criteo, SUM_L['criteo']),
    ('Kaggle', kaggle, SUM_L['kaggle']),
    ('Taobao', taobao, SUM_L['taobao']),
]

line_order_10 = [
    ('cps',  3),   # RAPID drawn first (bottom)
    ('acp',  4),   # ACP middle
    ('uniq', 5),   # Unique dashed on top (highest zorder)
]

for idx, (title, data, sum_l) in enumerate(datasets_fig10):
    ax = axes[idx]
    raw = B * sum_l

    for key, z in line_order_10:
        s = STYLE[key]
        vals = [(1 - data[g][key] / raw) for g in GPUS]
        ax.plot(X_POS, vals, marker=s['marker'], color=s['color'],
                linestyle=s['ls'], label=s['label'], zorder=z, markersize=6,
                fillstyle=s.get('fillstyle', 'full'), markeredgewidth=1.5)

    ax.set_xlabel('Cluster Size')
    if idx == 0:
        ax.set_ylabel('Deduplication Rate')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(X_POS)
    ax.set_xticklabels(GPU_LABELS)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.25, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Shared legend at top — reorder: Unique, ACP, RAPID
handles, labels = axes[0].get_legend_handles_labels()
handles = [handles[2], handles[1], handles[0]]
labels = [labels[2], labels[1], labels[0]]
fig.legend(handles, labels, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=8)

plt.tight_layout(w_pad=1.2)
fig10_path = os.path.join(OUT_DIR, 'deduplication_rate_improved_visibility4.pdf')
plt.savefig(fig10_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig10_path}")


# ============================================================
# Figure 11: Communication Volume
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))

datasets_fig11 = [
    ('DLRM-Criteo', criteo, 1e6, r'$\times 10^6$'),
    ('DCN-Kaggle', kaggle, 1e6, r'$\times 10^6$'),
    ('DeepFM-Taobao', taobao, 1e3, r'$\times 10^3$'),
]

line_order_11 = [
    ('red',  3),   # All-Reduction
    ('cps',  4),   # RAPID
    ('acp',  5),   # ACP
    ('uniq', 6),   # Unique on top (dashed, highest zorder)
]

for idx, (title, data, scale, scale_label) in enumerate(datasets_fig11):
    ax = axes[idx]

    for key, z in line_order_11:
        s = STYLE[key]
        vals = [data[g][key] / scale for g in GPUS]
        ax.plot(X_POS, vals, marker=s['marker'], color=s['color'],
                linestyle=s['ls'], label=s['label'], zorder=z, markersize=6,
                fillstyle=s.get('fillstyle', 'full'), markeredgewidth=1.5)

    ax.set_xlabel('Cluster Size')
    ax.set_ylabel(f'Comm Volume ({scale_label})')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(X_POS)
    ax.set_xticklabels(GPU_LABELS)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.25, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
# Reorder legend: All-Reduction, Unique, ACP, RAPID
handles = [handles[0], handles[3], handles[2], handles[1]]
labels = [labels[0], labels[3], labels[2], labels[1]]
fig.legend(handles, labels, loc='upper center', ncol=4,
           bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=8)

plt.tight_layout(w_pad=1.2)
fig11_path = os.path.join(OUT_DIR, 'comm_vol_final3.pdf')
plt.savefig(fig11_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig11_path}")
