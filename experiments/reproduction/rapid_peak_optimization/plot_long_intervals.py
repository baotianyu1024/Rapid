"""Plot cumulative net savings and per-window peak changes on a fixed trace."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'long_interval_report'
rows = json.loads((OUT / 'window_comparison.json').read_text())
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for interval in [4, 8, 12, 16]:
    selected = [r for r in rows if r['interval_windows'] == interval]
    windows = [r['window'] for r in selected]
    line, = axes[0].plot(windows, [r['cumulative_net_GiB'] for r in selected],
                        marker='.', label=f'Every {interval} windows')
    events = [r for r in selected if r['migration_GiB'] > 0]
    axes[0].scatter([r['window'] for r in events], [r['cumulative_net_GiB'] for r in events],
                    color=line.get_color(), marker='x', s=65)
    axes[1].plot(windows, [100 * (1 - r['peak'] / r['baseline_peak']) for r in selected],
                 color=line.get_color(), marker='.')
axes[0].set_ylabel('Cumulative net savings (GiB)\nAdam + route migration cost deducted')
axes[0].legend(ncol=2)
axes[0].set_title('64 ranks, fixed 20-window day6 trace; crosses mark migration windows')
axes[1].set_ylabel('Peak communication reduction (%)\nrelative to no migration')
axes[1].set_xlabel('Incremental window (20 training steps each)')
for ax in axes:
    ax.axhline(0, color='black', linewidth=.8, linestyle='--')
    ax.grid(alpha=.2)
axes[1].set_xticks(range(1, 21))
fig.tight_layout()
for suffix in ['png', 'pdf']:
    fig.savefig(OUT / f'long_intervals.{suffix}', dpi=160)
