"""Preview proposed high-pass Gaussian noise added after bicubic x4 downsampling."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from high_frequency_noise import high_frequency_noise


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="00709.png")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports/swinir_s_ffhq500_x4_seed42_150k_20261118/high_frequency_noise_preview.png")
    args = parser.parse_args()
    if Path(args.image).name != args.image:
        parser.error("--image must be a filename.")

    source = ROOT / "dataset/ffhq_lightweight_x4/lr" / args.image
    with Image.open(source) as image:
        clean = np.asarray(image.convert("RGB"), dtype=np.float32)
    if clean.shape != (256, 256, 3):
        raise ValueError("Expected a 256x256 RGB LR image.")
    noise = high_frequency_noise(clean.shape, seed=42)
    levels = (0, 8, 16, 24)  # Standard deviation in 8-bit intensity units.
    x0, y0, x1, y1 = 52, 65, 195, 149
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for col, sigma in enumerate(levels):
        pixels = np.clip(np.rint(clean + sigma * noise), 0, 255).astype(np.uint8)
        axes[0, col].imshow(pixels)
        axes[0, col].set_title(f"Noise sigma = {sigma}/255", fontsize=12)
        axes[0, col].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                             fill=False, edgecolor="#F7E25B", lw=1.5))
        axes[1, col].imshow(pixels[y0:y1, x0:x1], interpolation="nearest")
        axes[1, col].set_title("Same eye region", fontsize=11)
        for row in range(2):
            axes[row, col].axis("off")
    fig.suptitle(f"FFHQ {args.image}: bicubic x4 LR + high-pass noise preview\n"
                 "Four proposed strengths, fixed seed 42; HR and the active training run are unchanged",
                 fontsize=15, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
