"""Generate the three reviewer-response figures from its Markdown tables."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Response_English_Revised.md"
OUTPUT = ROOT / "figures" / "response"
OUTPUT.mkdir(parents=True, exist_ok=True)

INK = "#1d3440"
GRID = "#dce5e8"
SLATE = "#7b8f98"
RAPID = "#087f75"
HUGECTR = "#bb6841"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": GRID,
        "svg.fonttype": "path",
        "savefig.facecolor": "white",
    }
)


def table(number):
    text = SOURCE.read_text(encoding="utf-8")
    match = re.search(rf"^\*\*Table {number}\.[^\n]*\*\*", text, re.M)
    if not match:
        raise ValueError(f"Table {number} not found")
    block = re.search(r"(?:^\|[^\n]*\n)+", text[match.end() :], re.M)
    if not block:
        raise ValueError(f"Table {number} has no Markdown rows")
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in block.group().splitlines()]
    return rows[0], rows[2:]


def number(value):
    return float(re.sub(r"[^\d.]", "", value))


def save(fig, stem):
    path = OUTPUT / f"{stem}.svg"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.13)
    fig.savefig(Path("/tmp") / f"rapid_response_{stem}.png", dpi=160, bbox_inches="tight", pad_inches=0.13)
    plt.close(fig)
    print(path)


def throughput():
    header, rows = table(1)
    methods = header[1:]
    by_name = {row[0]: row for row in rows}
    chosen = [
        ("Kaggle / DCN (dim 32)", "Kaggle / DCN, dim32"),
        ("Terabyte / DLRM-DCNv2 (dim 32)", "Terabyte / DLRM-DCNv2, dim32"),
        ("Taobao / DeepFM (dim 128)", "Taobao / DeepFM, dim128"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.1), sharex=True)
    fig.subplots_adjust(hspace=0.40, left=0.19, right=0.98, top=0.91, bottom=0.10)
    fig.suptitle("Eight-GPU training throughput on the three main models", x=0.5, y=0.985, fontsize=12, weight="bold")
    for ax, (title, key) in zip(axes, chosen):
        values = [number(x) for x in by_name[key][1:]]
        ratio = values[methods.index("RAPID")] / values[methods.index("HugeCTR")]
        for index, (method, value) in enumerate(zip(methods, values)):
            color = RAPID if method == "RAPID" else HUGECTR if method == "HugeCTR" else SLATE
            ax.scatter(value, index, s=78 if method == "RAPID" else 43, color=color, zorder=3)
        ax.set_yticks(range(len(methods)), labels=methods)
        for label in ax.get_yticklabels():
            if label.get_text() == "RAPID":
                label.set_color(RAPID)
                label.set_weight("bold")
            elif label.get_text() == "HugeCTR":
                label.set_color(HUGECTR)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(8_000, 12_000_000)
        ax.set_ylim(len(methods) - 0.3, -0.7)
        ax.xaxis.set_major_locator(FixedLocator([10_000, 100_000, 1_000_000, 10_000_000]))
        ax.xaxis.set_major_formatter(FixedFormatter(["10k", "100k", "1M", "10M"]))
        ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.8, zorder=0)
        ax.set_title(f"{title}   |   RAPID / HugeCTR: {ratio:.2f}x", loc="left", fontsize=9.5, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, pad=4)
    axes[-1].set_xlabel("Samples/s (logarithmic scale)", labelpad=7)
    save(fig, "main_model_throughput")


def incremental():
    _, rows = table(3)
    labels = [row[0] for row in rows]
    values = [number(row[2]) for row in rows]
    colors = [SLATE, "#aec0c4", RAPID, "#466574"]
    fig, ax = plt.subplots(figsize=(7.0, 3.05))
    fig.subplots_adjust(left=0.26, right=0.98, top=0.84, bottom=0.20)
    ax.barh(range(len(rows)), values, color=colors, height=0.59)
    ax.set_yticks(range(len(rows)), labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 560_000)
    ax.set_xticks([0, 100_000, 200_000, 300_000, 400_000, 500_000], labels=["0", "100k", "200k", "300k", "400k", "500k"])
    ax.set_xlabel("Forward remote vectors received / rank / step", labelpad=7)
    for index, value in enumerate(values):
        ax.text(value + 7_000, index, f"{value / 1000:,.1f}k", va="center", fontsize=8.5)
    ax.set_title("Incremental placement retains most of the full-plan benefit", loc="left", fontsize=11, weight="bold", pad=12)
    ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=5)
    save(fig, "incremental_traffic")


def batch_size():
    _, rows = table(4)
    labels = [row[0] for row in rows]
    savings = [number(row[3]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.7, 3.15))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.82, bottom=0.22)
    ax.plot(range(len(rows)), savings, color=RAPID, linewidth=2.4, marker="o", markersize=6)
    ax.set_xticks(range(len(rows)), labels=labels)
    ax.set_ylim(73, 91)
    ax.set_yticks([75, 80, 85, 90])
    ax.set_ylabel("Remote-traffic reduction vs all Unique (%)", labelpad=7)
    ax.set_xlabel("Local batch size", labelpad=7)
    ax.set_title("RAPID's communication benefit grows with batch size", loc="left", fontsize=11, weight="bold", pad=12)
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8)
    for index in [0, len(rows) - 1]:
        ax.annotate(f"{savings[index]:.2f}%", (index, savings[index]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5, weight="bold", color=RAPID)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=5)
    save(fig, "batch_size_sensitivity")


if __name__ == "__main__":
    throughput()
    incremental()
    batch_size()
