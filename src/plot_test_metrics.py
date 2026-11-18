"""Plot all eight freshly reevaluated FFHQ-500 test metrics for three runs."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/three_runs_convergence/test_reevaluation/results.json"
OUTPUT = ROOT / "reports/three_runs_convergence/test_metrics_comparison.png"
BAR_OUTPUT = ROOT / "reports/three_runs_convergence/test_metrics_bar_comparison.png"
METRICS = (
    ("psnr", "PSNR (dB)", "higher"),
    ("ssim", "SSIM", "higher"),
    ("lpips", "LPIPS", "lower"),
    ("dists", "DISTS", "lower"),
    ("ms_ssim", "MS-SSIM", "higher"),
    ("fsim", "FSIM", "higher"),
    ("vifp", "VIFp", "higher"),
    ("uqi", "UQI", "higher"),
)
RUN_STYLE = {
    "bicubic": ("Bicubic only", "#2363A0"),
    "highfreq": ("High-frequency noise", "#C27624"),
    "masked": ("Masked noise", "#13877B"),
}


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    runs = data["runs"]
    if [run["key"] for run in runs] != list(RUN_STYLE):
        raise ValueError("Expected the three reevaluated runs in a fixed order.")
    if any(run["test_images"] != 50 for run in runs):
        raise ValueError("Expected 50 test images per run.")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for ax, (key, title, direction) in zip(axes.flat, METRICS):
        values = [run["metrics"][key] for run in runs]
        spread = max(values) - min(values)
        padding = max(spread * 0.20, abs(sum(values) / 3) * 0.003, 0.0001)
        for index, (run, value) in enumerate(zip(runs, values)):
            label, color = RUN_STYLE[run["key"]]
            position = 2 - index
            ax.scatter(value, position, s=95, color=color, zorder=3,
                       edgecolors="white", linewidths=0.8)
            ax.text(value + padding * 0.20, position,
                    f"{value:.3f}" if key == "psnr" else f"{value:.4f}",
                    va="center", fontsize=9, color=color, fontweight="bold")
        ax.set(xlim=(min(values) - padding, max(values) + padding * 2.7),
               ylim=(-0.5, 2.5), yticks=[2, 1, 0],
               yticklabels=[RUN_STYLE[run["key"]][0] for run in runs],
               title=f"{title}  {'↑' if direction == 'higher' else '↓'}")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.grid(axis="x", alpha=0.2)
    legend = [Line2D([0], [0], marker="o", linestyle="", markersize=9,
                     color=color, label=label) for label, color in RUN_STYLE.values()]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3, frameon=False)
    fig.suptitle("FFHQ-500 test set | 50 images per run | Eight metrics",
                 fontsize=18, fontweight="bold", y=0.995)
    fig.text(0.5, 0.025,
             "Dots show mean test scores; each panel uses its own x-axis scale. "
             "Input degradations differ across runs.",
             ha="center", color="#4A5360", fontsize=10)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.10,
                        wspace=0.62, hspace=0.45)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170)
    plt.close(fig)
    print(OUTPUT)

    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    bar_labels = ("Bicubic", "High freq.", "Masked")
    colors = [RUN_STYLE[run["key"]][1] for run in runs]
    for ax, (key, title, direction) in zip(axes.flat, METRICS):
        values = [run["metrics"][key] for run in runs]
        bars = ax.bar(range(3), values, color=colors, width=0.62)
        ax.set(title=f"{title}  {'↑' if direction == 'higher' else '↓'}",
               ylim=(0, max(values) * 1.17), xticks=range(3),
               xticklabels=bar_labels)
        ax.bar_label(bars, labels=[f"{v:.3f}" if key == "psnr" else f"{v:.4f}"
                                   for v in values], padding=3, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("FFHQ-500 test set | 50 images per run | Eight metrics",
                 fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.025,
             "All bar axes start at zero. Arrows show score direction; "
             "input degradations differ across runs.",
             ha="center", color="#4A5360", fontsize=10)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.10,
                        wspace=0.27, hspace=0.42)
    fig.savefig(BAR_OUTPUT, dpi=170)
    plt.close(fig)
    print(BAR_OUTPUT)


if __name__ == "__main__":
    main()
