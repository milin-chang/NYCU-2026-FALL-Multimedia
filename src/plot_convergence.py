"""Plot a snapshot of training loss and eight validation metrics for one run."""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
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


def read_training_events(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # The training process may still be writing the last line.
            if event.get("event") == "train":
                rows.append((event["step"], event["loss"]))
    return np.asarray(rows, dtype=float)


def read_learning_rate_events(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "train":
                rows.append((event["step"], event["lr"]))
    return np.asarray(rows, dtype=float)


def read_validation(run):
    rows = []
    for path in sorted((run / "validation").glob("step_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append((int(path.parent.name.removeprefix("step_")), summary))
    return rows


def read_per_image_validation(run, steps):
    records = []
    expected_images = None
    for step in steps:
        path = run / "validation" / f"step_{step:06d}" / "per_image.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        images = [row["image"] for row in rows]
        if expected_images is None:
            expected_images = set(images)
        if len(images) != 50 or set(images) != expected_images:
            raise ValueError(f"Expected the same 50 validation images in {path}")
        records.append(rows)
    return records


def plot_learning_rate(name, run, spec, output_dir):
    values = read_learning_rate_events(run / "events.jsonl")
    if len(values) == 0:
        raise ValueError("No learning-rate events found.")
    total = int(spec["train"]["steps"])
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    steps = np.r_[0, values[:, 0]]
    rates = np.r_[spec["train"]["learning_rate"], values[:, 1]]
    ax.step(steps, rates, where="post", lw=2.5, color="#195E99",
            label="Recorded learning rate")
    for milestone in spec["train"]["milestones"]:
        ax.axvline(milestone, color="#8C939C", ls="--", lw=0.9)
    ax.set(xlim=(0, total), ylim=(0, spec["train"]["learning_rate"] * 1.15),
           xlabel="Training step", ylabel="Learning rate",
           title=f"Learning rate | {name}")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.text(0.99, 0.95, f"Logged through step {int(values[-1, 0]):,}",
            transform=ax.transAxes, ha="right", va="top")
    fig.savefig(output_dir / "learning_rate.png", dpi=160)
    plt.close(fig)


def plot_per_image(name, run, validation, output_dir):
    steps = [step for step, _ in validation]
    records = read_per_image_validation(run, steps)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for ax, (metric, label, direction) in zip(axes.flat, METRICS):
        values = [np.array([float(row[metric]) for row in rows]) for rows in records]
        color = "#178C81" if direction == "higher" else "#B4652B"
        positions = np.arange(len(steps))
        ax.boxplot(values, positions=positions, widths=0.48, showfliers=False,
                   patch_artist=True,
                   boxprops={"facecolor": color, "alpha": 0.18, "edgecolor": color},
                   medianprops={"color": color, "linewidth": 2},
                   whiskerprops={"color": color}, capprops={"color": color})
        for x, scores in zip(positions, values):
            jitter = rng.uniform(-0.19, 0.19, len(scores))
            ax.scatter(x + jitter, scores, s=8, color=color, alpha=0.35,
                       edgecolors="none")
        ax.set(title=f"{label} ({direction} is better)",
               xticks=positions, xticklabels=[f"{step // 1000}k" for step in steps])
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(f"Per-image validation scores | {name}\n"
                 "Each dot = one of the same 50 FFHQ faces; box = middle 50%; line = median",
                 fontsize=16, fontweight="bold")
    fig.savefig(output_dir / "per_image_validation.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        parser.error("Invalid run name.")
    run = ROOT / "runs" / args.name
    spec = json.loads((ROOT / "spec" / f"{args.name}.json").read_text(encoding="utf-8"))
    training = read_training_events(run / "events.jsonl")
    validation = read_validation(run)
    if len(training) == 0 or len(validation) == 0:
        parser.error("Need at least one training log and validation result.")

    current_step = int(training[-1, 0])
    total_steps = int(spec["train"]["steps"])
    steps = training[:, 0]
    losses = training[:, 1]
    smoothed = np.array([np.median(losses[max(0, i - 9):i + 1])
                         for i in range(len(losses))])
    val_steps = np.array([step for step, _ in validation])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(17, 13), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[1.25, 1, 1, 1])
    ax_loss = fig.add_subplot(grid[0, :])
    ax_loss.plot(steps, losses, color="#A8BDD5", lw=0.9, alpha=0.7,
                 label="Single logged batch (every 500 steps)")
    ax_loss.plot(steps, smoothed, color="#195E99", lw=2.3,
                 label="Rolling median (10 logged batches)")
    ax_loss.set(title="Training L1 loss", xlabel="Training step", ylabel="L1 loss (log scale)",
                xlim=(0, total_steps))
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.23)
    ax_loss.legend(frameon=False, loc="upper right")

    for index, (key, label, direction) in enumerate(METRICS):
        ax = fig.add_subplot(grid[1 + index // 3, index % 3])
        scores = np.array([item["metrics"][key] for _, item in validation])
        color = "#178C81" if direction == "higher" else "#B4652B"
        ax.plot(val_steps, scores, "o-", color=color, lw=2, ms=5)
        ax.set(title=f"{label} ({direction} is better)", xlabel="Training step",
               xlim=(0, total_steps))
        ax.grid(alpha=0.23)
        ax.annotate(f"{scores[-1]:.4f}", xy=(val_steps[-1], scores[-1]),
                    xytext=(7, 2), textcoords="offset points", color=color,
                    fontsize=9, fontweight="bold")
        spread = max(float(np.ptp(scores)), abs(float(scores[-1])) * 0.005, 0.0001)
        ax.set_ylim(float(np.min(scores) - spread * 0.22),
                    float(np.max(scores) + spread * 0.35))

    ax_note = fig.add_subplot(grid[3, 2])
    ax_note.axis("off")
    status_path = run / "status.json"
    completed = (status_path.is_file() and
                 json.loads(status_path.read_text(encoding="utf-8")).get("state") == "complete")
    ax_note.text(0, 0.95, "Snapshot", va="top", fontsize=15, fontweight="bold")
    ax_note.text(0, 0.76,
                 f"Training: {current_step:,} / {total_steps:,} steps\n"
                 f"Validation: {len(validation)} points, 50 images each\n"
                 f"Latest validation: step {val_steps[-1]:,}\n"
                 "Metrics are evaluation only.\n" +
                 ("Final test completed." if completed else
                  "Final test is pending until training ends."),
                 va="top", linespacing=1.75)
    fig.suptitle(f"Lightweight SwinIR-S x4 on FFHQ-500 | {args.name}\n"
                 f"Snapshot {datetime.now():%Y-%m-%d %H:%M} (local time)",
                 fontsize=17, fontweight="bold")
    output = ROOT / "reports" / args.name / "convergence.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    plot_learning_rate(args.name, run, spec, output.parent)
    plot_per_image(args.name, run, validation, output.parent)
    print(f"Saved three plots in {output.parent}; current step {current_step:,}; "
          f"validations {len(validation)}")


if __name__ == "__main__":
    main()
