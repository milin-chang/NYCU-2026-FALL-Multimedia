"""Preview masked noise and intensity mean shift on an existing bicubic LR face."""

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
from masked_noise import soft_face_mask


ROOT = Path(__file__).resolve().parents[1]


def uint8_image(pixels):
    return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="00709.png")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports/swinir_s_ffhq500_x4_seed42_150k_20261118/additional_degradations_preview.png")
    args = parser.parse_args()
    if Path(args.image).name != args.image:
        parser.error("--image must be a filename.")
    with Image.open(ROOT / "dataset/ffhq_lightweight_x4/lr" / args.image) as image:
        clean = np.asarray(image.convert("RGB"), dtype=np.float32)
    if clean.shape != (256, 256, 3):
        raise ValueError("Expected a 256x256 RGB LR image.")

    high_frequency = 8 * high_frequency_noise(clean.shape, seed=42)
    mask = soft_face_mask(*clean.shape[:2])[..., None]
    masked_noise = 16 * np.random.default_rng(43).standard_normal(clean.shape) * mask
    shift = -30
    variants = [
        ("Bicubic LR only", uint8_image(clean)),
        ("High-frequency noise, 8/255", uint8_image(clean + high_frequency)),
        ("Masked noise, 16/255", uint8_image(clean + masked_noise)),
        ("Mean shift, -30/255", uint8_image(clean + shift)),
        ("All three combined", uint8_image(clean + high_frequency + masked_noise + shift)),
    ]
    x0, y0, x1, y1 = 52, 65, 195, 149
    fig, axes = plt.subplots(2, len(variants), figsize=(19, 7.5), constrained_layout=True)
    for col, (title, pixels) in enumerate(variants):
        axes[0, col].imshow(pixels)
        axes[0, col].set_title(title, fontsize=11)
        axes[0, col].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                             fill=False, edgecolor="#F7E25B", lw=1.5))
        axes[1, col].imshow(pixels[y0:y1, x0:x1], interpolation="nearest")
        axes[1, col].set_title("Same eye region", fontsize=10)
        for row in range(2):
            axes[row, col].axis("off")
    fig.suptitle(f"FFHQ {args.image}: proposed LR degradation variants\n"
                 "Mask = soft central face ellipse; fixed seeds; active training data unchanged",
                 fontsize=15, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
