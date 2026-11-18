"""Compare convergence of the completed FFHQ-500 bicubic and two noise runs."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from plot_convergence import (METRICS, ROOT, read_learning_rate_events,
                              read_training_events, read_validation)


RUNS = (
    ("swinir_s_ffhq500_x4_seed42_150k_20261118", "Bicubic only (150k)", "#2363A0"),
    ("swinir_s_ffhq500_x4_highfreq_s8_seed42_100k_20261118",
     "High-frequency noise, sigma 8 (100k)", "#C27624"),
    ("swinir_s_ffhq500_x4_masked_s16_seed42_100k_20261118",
     "Masked noise, sigma 16 (100k)", "#13877B"),
)
OUTPUT = ROOT / "reports" / "three_runs_convergence"


def step_label(value, _position):
    return "0" if value == 0 else f"{int(value / 1000)}k"


def load_runs():
    records = []
    for name, label, color in RUNS:
        run = ROOT / "runs" / name
        spec = json.loads((ROOT / "spec" / f"{name}.json").read_text(encoding="utf-8"))
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            raise ValueError(f"Run has not completed: {name}")
        training = read_training_events(run / "events.jsonl")
        learning_rates = read_learning_rate_events(run / "events.jsonl")
        validation = read_validation(run)
        if (len(training) == 0 or len(validation) == 0
                or int(training[-1, 0]) != spec["train"]["steps"]):
            raise ValueError(f"Incomplete training or validation logs: {name}")
        records.append({"name": name, "label": label, "color": color,
                        "spec": spec, "training": training,
                        "learning_rates": learning_rates, "validation": validation})
    return records


def draw_convergence(records):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for record in records:
        values = record["training"]
        losses = values[:, 1]
        smooth = np.array([np.median(losses[max(0, i - 9):i + 1])
                           for i in range(len(losses))])
        axes[0, 0].plot(values[:, 0], smooth, color=record["color"], lw=2.2,
                        label=record["label"])
        for ax, (key, _title, _direction) in zip(
                [axes.flat[i] for i in range(1, 9)], METRICS):
            validation = record["validation"]
            steps = [step for step, _ in validation]
            scores = [item["metrics"][key] for _, item in validation]
            ax.plot(steps, scores, marker="o", ms=3.5, lw=1.8,
                    color=record["color"])

    axes[0, 0].set(title="Training L1 loss (10-point rolling median; lower is better)",
                   ylabel="L1 loss")
    axes[0, 0].set_yscale("log")
    for ax, (_key, title, direction) in zip(
            [axes.flat[i] for i in range(1, 9)], METRICS):
        arrow = "↑" if direction == "higher" else "↓"
        ax.set_title(f"{title} ({arrow})")
    for ax in axes.flat:
        ax.set(xlim=(0, 150000), xlabel="Training step")
        ax.xaxis.set_major_formatter(FuncFormatter(step_label))
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.947),
               ncol=3, frameon=False)
    fig.suptitle("FFHQ-500 | Lightweight SwinIR-S x4 | Three training runs",
                 fontsize=18, fontweight="bold", y=0.995)
    fig.text(0.5, 0.018,
             "All curves use training step on the x-axis. Each validation point averages 50 faces. "
             "Noise conditions differ, so scores measure distinct tasks.",
             ha="center", fontsize=10, color="#4A5360")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.08,
                        hspace=0.52, wspace=0.27)
    target = OUTPUT / "convergence_comparison.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def draw_learning_rates(records):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for record in records:
        values = record["learning_rates"]
        steps = np.r_[0, values[:, 0]]
        rates = np.r_[record["spec"]["train"]["learning_rate"], values[:, 1]]
        ax.step(steps, rates, where="post", lw=2.2, color=record["color"],
                ls="--" if "masked" in record["name"] else "-",
                label=record["label"])
    ax.set(xlim=(0, 150000), xlabel="Training step", ylabel="Learning rate",
           title="Learning-rate schedules for the three runs")
    ax.xaxis.set_major_formatter(FuncFormatter(step_label))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    target = OUTPUT / "learning_rate_comparison.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = load_runs()
    for path in (draw_convergence(records), draw_learning_rates(records)):
        print(path)


if __name__ == "__main__":
    main()
