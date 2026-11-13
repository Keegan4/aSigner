"""Draw loss and macro-F1 per epoch from the history pretrain.py saves.

    pip install matplotlib
    python plot_pretrain.py runs/pretrained.json runs/pretrained.png

Two panels share the epoch axis (loss above, macro-F1 below) instead of one
chart with two y-axes. Each line keeps its colour in both panels.
"""
import argparse
import json
from pathlib import Path

# Training is measured on that epoch's augmented clips with dropout on, so it can sit below held-out.
SERIES = (
    ("train", "Training (augmented)", "#2a78d6"),
    ("asl-now", "ASLNow! held-out", "#eb6834"),
    ("casl-w60", "CASL-W60 test", "#1baf7a"),
    ("wlasl", "WLASL held-out", "#6250d6"),
)
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#bdbcb6", "#fcfcfb"


def plot(history_path: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(history_path.read_text())
    history = report["history"]
    epochs = [row["epoch"] for row in history]
    series = [s for s in SERIES if s[0] in history[0]]

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True, gridspec_kw={"hspace": 0.2})
    fig.patch.set_facecolor(SURFACE)
    for ax, metric, title in ((top, "loss", "Cross-entropy loss (label smoothing 0.1)"),
                              (bottom, "f1", "Macro-F1")):
        ax.set_facecolor(SURFACE)
        ends = []
        for key, label, colour in series:
            values = [row[key][metric] for row in history]
            ax.plot(epochs, values, color=colour, lw=2, label=label)
            ends.append([values[-1], values[-1]])
        # Direct end labels, nudged apart so they never overlap.
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        ends.sort(key=lambda e: e[0])
        for lower, upper in zip(ends, ends[1:]):
            upper[1] = max(upper[1], lower[1] + 0.06 * span)
        for value, placed in ends:
            ax.annotate(f"{value:.3f}", (epochs[-1], value), xytext=(epochs[-1] + 1.5, placed),
                        textcoords="data", va="center", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
    bottom.set_ylim(0, 1.02)
    bottom.set_xlim(1, epochs[-1] + max(6, len(epochs) // 10))
    bottom.set_xlabel("Epoch")
    top.legend(loc="upper right", frameon=False)
    names = [label.split(" held-out")[0].split(" test")[0] for key, label, _ in series if key != "train"]
    fig.suptitle(f"Pretraining on {' + '.join(names)}, {len(history)} epochs",
                 x=0.125, y=0.995, ha="left", fontsize=12, fontweight="bold", color=INK)
    note = "ASLNow! held-out shares signers with training (optimistic). "
    if "casl-w60" in history[0]:
        note += "CASL-W60 uses the dataset's own test split. "
    if "wlasl" in history[0]:
        note += "WLASL holds out 15% of each word. "
    fig.text(0.125, 0.947, note + "Training is on augmented clips with dropout on.", fontsize=8, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("history", type=Path, nargs="?", default=here / "runs" / "pretrained.json")
    parser.add_argument("out", type=Path, nargs="?", default=here / "runs" / "pretrained.png")
    args = parser.parse_args()
    plot(args.history, args.out)
