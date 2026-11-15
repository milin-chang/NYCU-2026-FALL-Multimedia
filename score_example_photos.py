"""Score FFHQ example photos against a reproducible bicubic x4 baseline.

Run from this directory with .venv/Scripts/python.exe score_example_photos.py
The baseline is interpolation, not SwinIR. All full-reference scores compare
the reconstructed image to the unmodified 1024x1024 FFHQ PNG.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
from importlib.metadata import version

import numpy as np
from PIL import Image
import torch
import piq
import pyiqa
from pypiqe import piqe
from sewar.full_ref import uqi


METRICS_DIR = Path(__file__).resolve().parent
SOURCE_DIR = METRICS_DIR / "example_photo"
OUTPUT_ROOT = METRICS_DIR / "outputs" / "example_photo"
torch.set_num_threads(min(8, os.cpu_count() or 1))


def as_tensor(image: Image.Image) -> torch.Tensor:
    data = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0)


def score(name, direction, function, scores, reference="HR vs bicubic x4"):
    try:
        with torch.inference_mode():
            value = function()
        if isinstance(value, (tuple, list)):
            value = value[0]
        if isinstance(value, torch.Tensor):
            value = value.item()
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("non-finite score")
        scores.append(dict(metric=name, score=value, direction=direction, images=reference, status="ok"))
        print(f"{name}: {value:.6f}", flush=True)
    except Exception as exc:
        scores.append(dict(metric=name, score="", direction=direction, images=reference, status=f"error: {exc}"))
        print(f"{name}: ERROR {exc}", flush=True)


def score_one(source: Path):
    output = OUTPUT_ROOT / f"{source.stem}_bicubic_x4"
    output.mkdir(parents=True, exist_ok=True)
    print(f"Scoring {source.name}", flush=True)
    hr = Image.open(source).convert("RGB")
    if hr.size != (1024, 1024):
        raise ValueError(f"Expected 1024x1024 FFHQ image: {source}: {hr.size}")
    lr = hr.resize((256, 256), Image.Resampling.BICUBIC)
    sr = lr.resize(hr.size, Image.Resampling.BICUBIC)
    lr.save(output / "lr_256.png")
    sr.save(output / "bicubic_1024.png")
    restored_tensor, reference_tensor = as_tensor(sr), as_tensor(hr)
    original, restored = np.asarray(hr), np.asarray(sr)
    scores = []

    score("PSNR", "higher", lambda: piq.psnr(restored_tensor, reference_tensor), scores)
    score("SSIM", "higher", lambda: piq.ssim(restored_tensor, reference_tensor), scores)
    score("MS-SSIM", "higher", lambda: piq.multi_scale_ssim(restored_tensor, reference_tensor), scores)
    score("FSIM", "higher", lambda: piq.fsim(restored_tensor, reference_tensor, chromatic=False), scores)
    score("FSIMc", "higher", lambda: piq.fsim(restored_tensor, reference_tensor, chromatic=True), scores)
    score("GMSD", "lower", lambda: piq.gmsd(restored_tensor, reference_tensor), scores)
    score("MDSI", "lower", lambda: piq.mdsi(restored_tensor, reference_tensor), scores)
    score("VIFp", "higher", lambda: piq.vif_p(restored_tensor, reference_tensor), scores)
    score("SR-SIM", "higher", lambda: piq.srsim(restored_tensor, reference_tensor), scores)
    score("VSI", "higher", lambda: piq.vsi(restored_tensor, reference_tensor), scores)
    score("UQI", "higher", lambda: uqi(original, restored), scores)
    score("PIQE (original)", "lower", lambda: piqe(original)[0], scores, "HR only")
    score("PIQE (bicubic x4)", "lower", lambda: piqe(restored)[0], scores, "bicubic x4 only")
    score("VIF", "higher", lambda: pyiqa.create_metric("vif", device="cpu")(restored_tensor, reference_tensor), scores)

    try:
        metric = pyiqa.create_metric("cw_ssim", device="cpu")
        score("CW-SSIM", "higher", lambda: metric(restored_tensor, reference_tensor), scores)
    except Exception as exc:
        scores.append(dict(metric="CW-SSIM", score="", direction="higher", images="HR vs bicubic x4", status=f"error: {exc}"))
        print(f"CW-SSIM: ERROR {exc}", flush=True)

    for name in ("IFC", "SPSIM", "RFSIM", "VSNR-FR", "GSM"):
        scores.append(dict(metric=name, score="", direction="higher", images="HR vs bicubic x4", status="not_computed: no verified implementation selected"))

    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "score", "direction", "images", "status"])
        writer.writeheader()
        writer.writerows(scores)
    (output / "run.json").write_text(
        json.dumps(
            {
                "source": str(source.relative_to(METRICS_DIR)).replace("\\", "/"),
                "source_size": list(hr.size),
                "scale": 4,
                "downsample": "Pillow BICUBIC 1024 -> 256",
                "upsample": "Pillow BICUBIC 256 -> 1024",
                "torch": torch.__version__,
                "piq": piq.__version__,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bicubic_sha256": hashlib.sha256((output / "bicubic_1024.png").read_bytes()).hexdigest(),
                "implementations": {
                    "PSNR, SSIM, MS-SSIM, FSIM, FSIMc, GMSD, MDSI, VIFp, SR-SIM, VSI": f"piq {version('piq')}",
                    "UQI": f"sewar {version('sewar')}",
                    "PIQE": f"pypiqe {version('pypiqe')}",
                    "VIF, CW-SSIM": f"pyiqa {version('pyiqa')}",
                },
                "scores": scores,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return [dict(image=source.name, **row) for row in scores]


def main():
    sources = sorted(SOURCE_DIR.glob("*.png"))
    if not sources:
        raise FileNotFoundError(f"No PNG images in {SOURCE_DIR}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_scores = []
    for source in sources:
        all_scores.extend(score_one(source))
    with (OUTPUT_ROOT / "scores_all.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "metric", "score", "direction", "images", "status"])
        writer.writeheader()
        writer.writerows(all_scores)


if __name__ == "__main__":
    main()
