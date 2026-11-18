"""Create fixed bicubic x4 plus face-region masked-noise LR PNGs for FFHQ-500."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from masked_noise import add_masked_noise


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "dataset/ffhq_split_seed42_500.json")
    parser.add_argument("--clean-lr", type=Path,
                        default=ROOT / "dataset/ffhq_lightweight_x4/lr")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "dataset/ffhq_lightweight_x4_masked_s16/lr")
    parser.add_argument("--sigma", type=float, default=16.0)
    args = parser.parse_args()
    if args.sigma <= 0:
        parser.error("Sigma must be positive.")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    names = sorted(name for split in ("train", "val", "test")
                   for name in manifest["splits"][split])
    if len(names) != 500 or len(set(names)) != 500:
        parser.error("Expected 500 unique FFHQ image names.")
    clean_metadata = args.clean_lr.parent / "metadata.json"
    if json.loads(clean_metadata.read_text(encoding="utf-8"))["manifest_sha256"] != sha256(args.manifest):
        parser.error("Clean bicubic LR metadata does not match the split manifest.")
    metadata = {
        "manifest_sha256": sha256(args.manifest),
        "clean_lr_metadata_sha256": sha256(clean_metadata),
        "method": "KAIR bicubic x4 antialiasing, then soft-face-mask Gaussian noise",
        "sigma_8bit": args.sigma,
        "mask": "soft ellipse centered at (0.50W,0.52H), radii (0.34W,0.36H), linear edge from squared radius 0.90 to 1.10",
        "per_image_seed_rule": "manifest seed + integer image stem",
        "train_noise": "fresh noise on each cropped 64x64 LR patch with the corresponding cropped and augmented full-image mask",
        "images": 500,
        "lr_size": [256, 256],
    }
    info = args.out.parent / "metadata.json"
    if info.exists() and json.loads(info.read_text(encoding="utf-8")) != metadata:
        parser.error(f"Existing metadata differs: {info}")
    args.out.mkdir(parents=True, exist_ok=True)
    created = 0
    for index, name in enumerate(names, 1):
        target = args.out / name
        if target.is_file():
            with Image.open(target) as existing:
                if existing.mode == "RGB" and existing.size == (256, 256):
                    continue
        with Image.open(args.clean_lr / name) as image:
            clean = image.convert("RGB")
            if clean.size != (256, 256):
                raise ValueError(f"Expected 256x256 clean bicubic LR: {name}")
            rgb8 = np.asarray(clean)
        noisy = add_masked_noise(rgb8, args.sigma,
                                 seed=manifest["seed"] + int(Path(name).stem))
        temporary = target.with_suffix(".part")
        Image.fromarray(noisy).save(temporary, format="PNG")
        temporary.replace(target)
        created += 1
        if index % 50 == 0 or index == len(names):
            print(f"Prepared {index}/{len(names)}; new files: {created}", flush=True)
    temporary = info.with_suffix(".part")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(info)
    print(f"Masked LR images: {args.out} (new: {created}, total: 500)")


if __name__ == "__main__":
    main()
