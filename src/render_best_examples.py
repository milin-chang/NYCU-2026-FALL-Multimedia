"""Run each FFHQ-500 best EMA checkpoint on one shared test face."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from benchmark_swinir import make_model


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("bicubic", "Bicubic only", "swinir_s_ffhq500_x4_seed42_150k_20261118"),
    ("highfreq", "High-frequency noise", "swinir_s_ffhq500_x4_highfreq_s8_seed42_100k_20261118"),
    ("masked", "Masked noise", "swinir_s_ffhq500_x4_masked_s16_seed42_100k_20261118"),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def labeled_tile(image, label, crop=None):
    if crop is not None:
        image = image.crop(crop)
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (512, 550), "white")
    tile.paste(image, (0, 38))
    draw = ImageDraw.Draw(tile)
    draw.text((14, 8), label, fill="#17212E", font=font(22))
    return tile


def mosaic(path, top, bottom=None, crop=None):
    columns = len(top)
    rows = 2 if bottom is not None else 1
    sheet = Image.new("RGB", (columns * 512, rows * 550), "#EEF1F4")
    for index, (image, label) in enumerate(top):
        sheet.paste(labeled_tile(image, label, crop), (index * 512, 0))
    if bottom is not None:
        for index, (image, label) in enumerate(bottom):
            sheet.paste(labeled_tile(image, label, crop), (index * 512, 550))
    sheet.save(path)


def infer(model, lr_rgb8, device):
    x = torch.from_numpy(lr_rgb8.copy()).permute(2, 0, 1).unsqueeze(0).to(
        device, dtype=torch.float32) / 255.0
    with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        y = model(x)
    return y.float().clamp(0, 1).mul(255).round().byte().squeeze(0).permute(
        1, 2, 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="00747.png",
                        help="Filename from the common FFHQ-500 test split")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "dataset/ffhq_split_seed42_500.json").read_text(
        encoding="utf-8"))
    if args.image not in manifest["splits"]["test"]:
        parser.error("The requested image is not in the FFHQ-500 test split.")
    output = ROOT / "reports" / "three_runs_convergence" / f"example_{Path(args.image).stem}"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    hr_path = ROOT / "dataset/ffhq_1024" / args.image
    hr = Image.open(hr_path).convert("RGB")
    hr.save(output / "hr_reference_1024.png")
    for key, label, name in RUNS:
        spec = json.loads((ROOT / "spec" / f"{name}.json").read_text(encoding="utf-8"))
        lr_path = Path(spec["data"]["lr_images"]) / args.image
        checkpoint = ROOT / "runs" / name / "checkpoints" / "best.pth"
        lr = Image.open(lr_path).convert("RGB")
        if lr.size != (256, 256) or hr.size != (1024, 1024):
            raise ValueError(f"Unexpected LR/HR dimensions for {name}")
        lr.save(output / f"{key}_input_256.png")
        model = make_model().eval().to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["ema"])
        sr_rgb8 = infer(model, np.asarray(lr), device)
        sr_path = output / f"{key}_sr_1024.png"
        Image.fromarray(sr_rgb8).save(sr_path)
        previous = ROOT / "runs" / name / "test" / "sr" / args.image
        with Image.open(previous) as image:
            if not np.array_equal(sr_rgb8, np.asarray(image.convert("RGB"))):
                raise ValueError(f"Fresh best.pth output differs from saved test SR: {name}")
        results.append({"key": key, "label": label, "run": name,
                        "checkpoint_step": state["step"],
                        "checkpoint_sha256": sha256(checkpoint),
                        "input": str(lr_path), "output": str(sr_path),
                        "matches_saved_test_output": True})
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Rendered {label}: {args.image}", flush=True)

    inputs = [Image.open(output / f"{key}_input_256.png").convert("RGB").resize(
        (1024, 1024), Image.Resampling.BICUBIC) for key, _label, _name in RUNS]
    outputs = [Image.open(output / f"{key}_sr_1024.png").convert("RGB")
               for key, _label, _name in RUNS]
    labels = [label for _key, label, _name in RUNS]
    mosaic(output / "input_vs_output.png",
           list(zip(inputs, [f"Input: {label}" for label in labels])),
           list(zip(outputs, [f"Output: {label}" for label in labels])))
    detail = (256, 256, 768, 768)
    mosaic(output / "face_detail.png",
           [(hr, "HR reference"), *zip(outputs, labels)], crop=detail)
    (output / "provenance.json").write_text(json.dumps({
        "test_image": args.image, "hr": str(hr_path),
        "inference_weights": "EMA state from each best.pth",
        "precision": "CUDA BF16" if device.type == "cuda" else "CPU FP32",
        "detail_crop_xyxy": detail, "runs": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved examples to {output}")


if __name__ == "__main__":
    main()
