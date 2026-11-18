"""Show the effective detail difference between one FFHQ LR/HR image pair."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="00709.png")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports/swinir_s_ffhq500_x4_seed42_150k_20261118/lr_hr_example.png")
    args = parser.parse_args()
    if Path(args.image).name != args.image:
        parser.error("--image must be a filename from the FFHQ split.")

    lr_path = ROOT / "dataset/ffhq_lightweight_x4/lr" / args.image
    hr_path = ROOT / "dataset/ffhq_1024" / args.image
    with Image.open(lr_path) as source, Image.open(hr_path) as target:
        lr, hr = source.convert("RGB"), target.convert("RGB")
    if lr.size != (256, 256) or hr.size != (1024, 1024):
        raise ValueError("Expected a 256x256 LR and 1024x1024 HR pair.")

    images = [
        np.asarray(lr.resize(hr.size, Image.Resampling.NEAREST)),
        np.asarray(lr.resize(hr.size, Image.Resampling.BICUBIC)),
        np.asarray(hr),
    ]
    titles = ["LR 256x256 (nearest display)",
              "LR 256x256 (bicubic display)", "Original HR 1024x1024"]
    # The same eye and hair region is shown below all three full images.
    x0, y0, x1, y1 = 260, 300, 750, 590
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    for col, (pixels, title) in enumerate(zip(images, titles)):
        axes[0, col].imshow(pixels)
        axes[0, col].set_title(title, fontsize=12)
        axes[0, col].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                             fill=False, edgecolor="#F7E25B", lw=2))
        axes[1, col].imshow(pixels[y0:y1, x0:x1])
        axes[1, col].set_title("Same eye region enlarged", fontsize=11)
        for row in range(2):
            axes[row, col].axis("off")
    fig.suptitle(f"FFHQ {args.image}: synthetic x4 LR versus original HR\n"
                 "Displaying LR larger cannot restore details removed during downsampling",
                 fontsize=16, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
