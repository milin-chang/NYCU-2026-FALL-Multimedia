"""Create fixed x4 MATLAB-style bicubic LR PNGs for the FFHQ 500-image split."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAIR_ROOT = Path(__file__).resolve().parent / "upstream/KAIR"
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
sys.path.insert(0, str(KAIR_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "dataset/ffhq_split_seed42_500.json")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "dataset/ffhq_1024")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "dataset/ffhq_lightweight_x4/lr")
    args = parser.parse_args()
    if not KAIR_ROOT.is_dir():
        parser.error("Missing src/upstream/KAIR. Run src/setup.ps1 first.")
    from utils.utils_image import imresize_np, single2uint

    manifest_bytes = args.manifest.read_bytes()
    splits = json.loads(manifest_bytes)["splits"]
    names = sorted(name for split in ("train", "val", "test") for name in splits[split])
    if len(names) != len(set(names)):
        parser.error("Split manifest contains duplicate image names.")
    metadata = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "kair_commit": "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e",
        "method": "utils_image.imresize_np, scale=0.25, antialiasing=True, single2uint",
        "images": len(names),
        "lr_size": [256, 256],
    }
    info = args.out.parent / "metadata.json"
    if info.exists() and json.loads(info.read_text(encoding="utf-8")) != metadata:
        parser.error(f"Existing LR metadata differs from this request: {info}")
    args.out.mkdir(parents=True, exist_ok=True)
    created = 0
    for index, name in enumerate(names, 1):
        target = args.out / name
        if target.is_file():
            with Image.open(target) as existing:
                if existing.mode == "RGB" and existing.size == (256, 256):
                    continue
        with Image.open(args.images / name) as raw:
            hr = ImageOps.exif_transpose(raw).convert("RGB")
            if hr.size != (1024, 1024):
                raise ValueError(f"Expected a 1024x1024 FFHQ image: {name}")
            pixels = np.asarray(hr, dtype=np.float32) / 255.0
        lr = imresize_np(pixels, 0.25, antialiasing=True)
        temporary = target.with_suffix(".part")
        Image.fromarray(single2uint(lr)).save(temporary, format="PNG")
        temporary.replace(target)
        created += 1
        if index % 50 == 0 or index == len(names):
            print(f"Prepared {index}/{len(names)}; new files: {created}", flush=True)
    contents = json.dumps(metadata, indent=2) + "\n"
    temporary = info.with_suffix(".part")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(info)
    print(f"LR images: {args.out} (new: {created}, total: {len(names)})")


if __name__ == "__main__":
    main()
