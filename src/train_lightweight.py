"""Train Lightweight SwinIR-S x4 on FFHQ and evaluate eight metrics separately."""

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR

from benchmark_swinir import KAIR_COMMIT, make_model, make_pair
from high_frequency_noise import high_frequency_noise
from masked_noise import soft_face_mask
from quality_eval import METRIC_DIRECTIONS, MetricSuite


ROOT = Path(__file__).resolve().parents[1]
METRICS = tuple(METRIC_DIRECTIONS)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def log_event(run, kind, **values):
    event = {"time_utc": now(), "event": kind, **values}
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def run_paths(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Run name must contain only letters, numbers, hyphens, or underscores.")
    return ROOT / "runs" / name, ROOT / "spec" / f"{name}.json", ROOT / "reports" / name


def load_data(manifest_path, images, lr_images):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    names = [name for split in ("train", "val", "test") for name in splits[split]]
    if len(names) != len(set(names)):
        raise ValueError("Split manifest has duplicate image names.")
    missing = [name for name in names
               if not (images / name).is_file() or not (lr_images / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} HR/LR pairs; first: {missing[0]}")
    lr_metadata = lr_images.parent / "metadata.json"
    if json.loads(lr_metadata.read_text(encoding="utf-8"))["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("LR images were prepared from a different split manifest.")
    return manifest, lr_metadata


def build_spec(args, manifest, lr_metadata):
    import DISTS_pytorch
    import lpips
    import piq
    import sewar

    spec = {
        "name": args.name,
        "created_utc": now(),
        "data": {
            "dataset": "FFHQ aligned RGB PNG",
            "images": str(args.images.resolve()),
            "lr_images": str(args.lr_images.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "lr_metadata_sha256": sha256(lr_metadata),
            "counts": manifest["counts"],
            "seed": manifest["seed"],
            "degradation": "KAIR imresize_np bicubic x4 with antialiasing; 8-bit LR PNG",
            "train_patch_lr": 64,
            "train_patch_hr": 256,
            "train_augment_modes": 8,
        },
        "model": {
            "name": "Lightweight SwinIR-S x4",
            "source_sha256": sha256(Path(__file__).with_name("network_swinir.py")),
            "kair_commit": KAIR_COMMIT,
            "rstb": 4,
            "stl_per_rstb": 6,
            "embed_dim": 60,
            "attention_heads": 6,
            "window_size": 8,
            "upsampler": "pixelshuffledirect",
            "initialization": "random, no pretrained generator weights",
        },
        "train": {
            "steps": args.steps,
            "batch": args.batch,
            "precision": args.precision,
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "scheduler": "MultiStepLR",
            "milestones": [args.steps // 2, args.steps * 4 // 5,
                           args.steps * 9 // 10, args.steps * 19 // 20],
            "gamma": 0.5,
            "loss": "L1 pixel only",
            "ema_decay": 0.999,
            "log_every": args.log_every,
            "save_every": args.save_every,
            "val_every": args.val_every,
            "val_limit": args.val_limit,
            "test_limit": args.test_limit,
            "best_checkpoint_by": "highest validation PSNR",
        },
        "evaluation": {
            "metrics": METRIC_DIRECTIONS,
            "image_format": "RGB uint8 PNG; metrics on saved SR pixels versus HR",
            "psnr_ssim_ms_ssim_fsim_vifp": f"piq=={piq.__version__}",
            "lpips": "lpips==0.1.4; AlexNet v0.1; inputs [-1,1]",
            "dists": "dists-pytorch==0.1; inputs [0,1]",
            "uqi": "sewar==0.4.10; uint8 RGB",
            "validation": f"{args.val_limit or 50} val images at each val interval",
            "test": f"{args.test_limit or 50} test images once, using best validation checkpoint",
            "metrics_used_for_gradient": False,
        },
        "software": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "piq": piq.__version__,
            "lpips_package": str(Path(lpips.__file__).name),
            "dists_package": str(Path(DISTS_pytorch.__file__).name),
            "sewar_package": str(Path(sewar.__file__).name),
        },
    }
    if args.noise_sigma > 0:
        spec["data"]["degradation"] = (
            "KAIR bicubic x4 antialiasing, then high-pass Gaussian noise "
            f"sigma={args.noise_sigma}/255 cutoff={args.noise_cutoff}; RGB uint8 PNG")
        spec["data"]["clean_lr_images"] = str(args.clean_lr_images.resolve())
        spec["data"]["train_noise"] = (
            "fresh high-pass Gaussian noise on each 64x64 LR patch, "
            "seed=manifest seed + 1000000 + sample position; quantized to uint8")
    elif args.masked_noise_sigma > 0:
        spec["data"]["degradation"] = (
            "KAIR bicubic x4 antialiasing, then central soft-face-mask Gaussian noise "
            f"sigma={args.masked_noise_sigma}/255; RGB uint8 PNG")
        spec["data"]["clean_lr_images"] = str(args.clean_lr_images.resolve())
        spec["data"]["train_noise"] = (
            "fresh Gaussian noise on the cropped and augmented full-image face mask, "
            "seed=manifest seed + 2000000 + sample position; quantized to uint8")
    return spec


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    torch.save(state, temporary)
    temporary.replace(path)


def checkpoint_state(step, best, model, ema, optimizer, scheduler):
    return {
        "step": step,
        "best_psnr": best,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def train_batch(paths, lr_images, step, batch, seed, device,
                noise_sigma=0.0, noise_cutoff=0.08, masked_noise_sigma=0.0):
    pairs = []
    for offset in range(batch):
        position = (step - 1) * batch + offset
        hr_path = paths[position % len(paths)]
        if masked_noise_sigma > 0:
            from utils.utils_image import augment_img

            lr, hr, x, y, mode = make_pair(hr_path, lr_images / hr_path.name,
                                           seed + position, return_transform=True)
            mask = soft_face_mask(256, 256)[y:y + 64, x:x + 64]
            mask = augment_img(mask, mode)
            clean = lr.permute(1, 2, 0).numpy()
            noise = np.random.default_rng(seed + 2_000_000 + position).standard_normal(clean.shape)
            noisy = np.clip(np.rint((clean + masked_noise_sigma / 255 * noise * mask[..., None]) * 255),
                            0, 255).astype(np.float32) / 255
            lr = torch.from_numpy(np.ascontiguousarray(noisy.transpose(2, 0, 1)))
        else:
            lr, hr = make_pair(hr_path, lr_images / hr_path.name, seed + position)
        if noise_sigma > 0:
            clean = lr.permute(1, 2, 0).numpy()
            noise = high_frequency_noise(clean.shape, seed=seed + 1_000_000 + position,
                                         cutoff=noise_cutoff)
            noisy = np.clip(np.rint((clean + noise_sigma / 255 * noise) * 255),
                            0, 255).astype(np.float32) / 255
            lr = torch.from_numpy(np.ascontiguousarray(noisy.transpose(2, 0, 1)))
        pairs.append((lr, hr))
    lr = torch.stack([pair[0] for pair in pairs]).to(device)
    hr = torch.stack([pair[1] for pair in pairs]).to(device)
    return lr, hr


@torch.inference_mode()
def evaluate(model, suite, names, images, lr_images, device, precision, output, save_sr=False):
    model.eval()
    if save_sr:
        (output / "sr").mkdir(parents=True, exist_ok=True)
    rows = []
    for index, name in enumerate(names, 1):
        with Image.open(lr_images / name) as source:
            lr_rgb8 = np.asarray(source.convert("RGB"))
        with Image.open(images / name) as target:
            hr_rgb8 = np.asarray(target.convert("RGB"))
        lr = torch.from_numpy(lr_rgb8.copy()).permute(2, 0, 1).unsqueeze(0).to(
            device, dtype=torch.float32) / 255.0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            sr = model(lr)
        sr_rgb8 = sr.float().clamp(0, 1).mul(255).round().byte().squeeze(0).permute(
            1, 2, 0).cpu().numpy()
        if sr_rgb8.shape != hr_rgb8.shape:
            raise ValueError(f"SR and HR shape mismatch for {name}")
        if save_sr:
            Image.fromarray(sr_rgb8).save(output / "sr" / name)
        values = suite.score(sr_rgb8, hr_rgb8)
        rows.append({"image": name, **values})
        if index % 10 == 0 or index == len(names):
            print(f"Evaluated {index}/{len(names)}: {output.name}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    average = {metric: statistics.mean(row[metric] for row in rows) for metric in METRICS}
    write_json(output / "summary.json", {"images": len(rows), "metrics": average})
    return average


def write_report(spec, validation, best_step, test_scores, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": spec["name"],
        "completed_utc": now(),
        "best_step": best_step,
        "best_by": "validation PSNR",
        "validation": validation,
        "test": test_scores,
        "metric_directions": METRIC_DIRECTIONS,
    }
    write_json(report_dir / "results.json", result)
    lines = [f"# {spec['name']}", "", "## 實驗規格", "",
             f"- 資料：FFHQ {sum(spec['data']['counts'].values())} 張；訓練／驗證／測試 = "
             f"{spec['data']['counts']['train']}／{spec['data']['counts']['val']}／{spec['data']['counts']['test']}",
             f"- 模型：Lightweight SwinIR-S ×4，隨機初始化，訓練 {spec['train']['steps']:,} 步",
             f"- 訓練損失：{spec['train']['loss']}；八個品質指標僅用於評分",
             f"- 驗證：{spec['evaluation']['validation']}；測試：{spec['evaluation']['test']}",
             f"- 最佳模型：第 {best_step:,} 步，按驗證 PSNR 選出", "",
             "## 測試集平均分數", "", "| 指標 | 分數 | 方向 |", "| --- | ---: | --- |"]
    for metric in METRICS:
        lines.append(f"| {metric.upper()} | {test_scores[metric]:.6f} | {METRIC_DIRECTIONS[metric]} |")
    lines += ["", "## 驗證過程", "", "| 步數 | PSNR | SSIM | LPIPS | DISTS | MS-SSIM | FSIM | VIFp | UQI |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in validation:
        scores = item["metrics"]
        lines.append("| " + str(item["step"]) + " | " + " | ".join(
            f"{scores[metric]:.4f}" for metric in METRICS) + " |")
    lines += ["", f"逐張分數與 SR 圖：`runs/{spec['name']}/test/`", ""]
    (report_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Shared name for spec/, runs/, and reports/")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "dataset/ffhq_split_seed42_500.json")
    parser.add_argument("--images", type=Path, default=ROOT / "dataset/ffhq_1024")
    parser.add_argument("--lr-images", type=Path,
                        default=ROOT / "dataset/ffhq_lightweight_x4/lr")
    parser.add_argument("--clean-lr-images", type=Path,
                        default=ROOT / "dataset/ffhq_lightweight_x4/lr")
    parser.add_argument("--noise-sigma", type=float, default=0.0,
                        help="High-frequency noise standard deviation in 8-bit pixel units")
    parser.add_argument("--masked-noise-sigma", type=float, default=0.0,
                        help="Face-region masked Gaussian noise standard deviation in 8-bit pixel units")
    parser.add_argument("--noise-cutoff", type=float, default=0.08,
                        help="High-pass filter cutoff in cycles per pixel")
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--val-every", type=int, default=15_000)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--val-limit", type=int, default=None,
                        help="Preflight only: evaluate the first N validation images")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Preflight only: evaluate the first N test images")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write a queued run spec without starting GPU training")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.lr, args.val_every, args.save_every, args.log_every) <= 0:
        parser.error("Steps, batch, intervals, and learning rate must be positive.")
    if args.val_limit is not None and not 1 <= args.val_limit <= 50:
        parser.error("--val-limit must be between 1 and 50.")
    if args.test_limit is not None and not 1 <= args.test_limit <= 50:
        parser.error("--test-limit must be between 1 and 50.")
    if (args.noise_sigma < 0 or args.masked_noise_sigma < 0
            or not 0 < args.noise_cutoff < 0.5):
        parser.error("Noise sigmas must be non-negative and cutoff between 0 and 0.5.")
    if args.noise_sigma > 0 and args.masked_noise_sigma > 0:
        parser.error("Choose one noise type per run.")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment.")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("BF16 is unavailable on this GPU; use --precision fp32.")
    run, spec_path, report_dir = run_paths(args.name)
    manifest, lr_metadata = load_data(args.manifest, args.images, args.lr_images)
    lr_details = json.loads(lr_metadata.read_text(encoding="utf-8"))
    if args.noise_sigma > 0:
        clean_metadata = args.clean_lr_images.parent / "metadata.json"
        if (lr_details.get("method") != "KAIR bicubic x4 antialiasing, then high-pass Gaussian noise"
                or lr_details.get("sigma_8bit") != args.noise_sigma
                or lr_details.get("cutoff_cycles_per_pixel") != args.noise_cutoff
                or lr_details.get("clean_lr_metadata_sha256") != sha256(clean_metadata)):
            parser.error("Noisy LR metadata does not match the requested high-frequency noise setup.")
        missing_clean = [name for name in manifest["splits"]["train"]
                         if not (args.clean_lr_images / name).is_file()]
        if missing_clean:
            parser.error(f"Missing clean training LR pair: {missing_clean[0]}")
    elif args.masked_noise_sigma > 0:
        clean_metadata = args.clean_lr_images.parent / "metadata.json"
        if (lr_details.get("method") != "KAIR bicubic x4 antialiasing, then soft-face-mask Gaussian noise"
                or lr_details.get("sigma_8bit") != args.masked_noise_sigma
                or lr_details.get("clean_lr_metadata_sha256") != sha256(clean_metadata)):
            parser.error("Masked LR metadata does not match the requested noise setup.")
        missing_clean = [name for name in manifest["splits"]["train"]
                         if not (args.clean_lr_images / name).is_file()]
        if missing_clean:
            parser.error(f"Missing clean training LR pair: {missing_clean[0]}")
    elif "sigma_8bit" in lr_details:
        parser.error("Noisy LR images require --noise-sigma and matching settings.")
    expected = build_spec(args, manifest, lr_metadata)
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in spec.items() if key != "created_utc"}
        expected_comparable = {key: value for key, value in expected.items() if key != "created_utc"}
        if comparable != expected_comparable or not args.resume:
            parser.error("Run spec already exists or arguments differ; use --resume with identical settings.")
    elif args.resume:
        parser.error("Cannot resume without an existing run spec.")
    else:
        write_json(spec_path, expected)
        spec = expected
    run.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        if args.resume:
            parser.error("--prepare-only cannot be combined with --resume.")
        write_json(run / "status.json", {"state": "queued", "step": 0,
                                           "total_steps": args.steps, "updated_utc": now()})
        log_event(run, "queued", step=0, total_steps=args.steps)
        return
    device = torch.device("cuda")
    seed = manifest["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model().train().to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=spec["train"]["milestones"], gamma=0.5)
    start, best = 0, -math.inf
    last = run / "checkpoints/last.pth"
    if args.resume:
        if last.is_file():
            state = torch.load(last, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            ema.load_state_dict(state["ema"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start, best = state["step"], state["best_psnr"]
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"])
            if start >= args.steps:
                parser.error(f"Checkpoint already reached step {start}.")
        elif json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "queued":
            parser.error("No checkpoint found for this run.")
    suite = MetricSuite(device)
    train_paths = [args.images / name for name in manifest["splits"]["train"]]
    train_lr_images = (args.clean_lr_images if args.noise_sigma > 0 or args.masked_noise_sigma > 0
                       else args.lr_images)
    random.Random(seed).shuffle(train_paths)
    log_event(run, "resume" if start else "start", step=start, total_steps=args.steps,
              gpu=torch.cuda.get_device_name(device), parameters=sum(p.numel() for p in model.parameters()))
    began = time.monotonic()
    validation = []
    for path in sorted((run / "validation").glob("step_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        validation.append({"step": int(path.parent.name.split("_")[1]), "metrics": summary["metrics"]})
    for step in range(start + 1, args.steps + 1):
        lr, hr = train_batch(train_paths, train_lr_images, step, args.batch, seed, device,
                             noise_sigma=args.noise_sigma, noise_cutoff=args.noise_cutoff,
                             masked_noise_sigma=args.masked_noise_sigma)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            sr = model(lr)
            loss = F.l1_loss(sr, hr)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite training loss at step {step}.")
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 0.001)
        loss_value = loss.item()
        del lr, hr, sr, loss
        if step == start + 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.monotonic() - began
            log_event(run, "train", step=step, loss=loss_value,
                      lr=optimizer.param_groups[0]["lr"], elapsed_s=round(elapsed, 2))
            write_json(run / "status.json", {"state": "training", "step": step,
                                               "total_steps": args.steps,
                                               "best_val_psnr": best if math.isfinite(best) else None,
                                               "elapsed_s": elapsed, "updated_utc": now()})
        if step % args.val_every == 0 or step == args.steps:
            output = run / "validation" / f"step_{step:06d}"
            scores = evaluate(ema, suite, manifest["splits"]["val"][:args.val_limit], args.images,
                              args.lr_images, device, args.precision, output)
            validation.append({"step": step, "metrics": scores})
            log_event(run, "validation", step=step, **scores)
            if scores["psnr"] > best:
                best = scores["psnr"]
                save_checkpoint(run / "checkpoints/best.pth",
                                {"step": step, "ema": ema.state_dict(), "validation": scores})
                log_event(run, "best", step=step, psnr=best)
            model.train()
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(last, checkpoint_state(step, best, model, ema, optimizer, scheduler))
            log_event(run, "checkpoint", step=step,
                      best_val_psnr=best if math.isfinite(best) else None)
    best_state = torch.load(run / "checkpoints/best.pth", map_location="cpu", weights_only=False)
    ema.load_state_dict(best_state["ema"])
    test_scores = evaluate(ema, suite, manifest["splits"]["test"][:args.test_limit], args.images,
                           args.lr_images, device, args.precision, run / "test", save_sr=True)
    write_report(spec, validation, best_state["step"], test_scores, report_dir)
    log_event(run, "complete", step=args.steps, best_step=best_state["step"], **test_scores)
    write_json(run / "status.json", {"state": "complete", "step": args.steps,
                                       "total_steps": args.steps, "best_step": best_state["step"],
                                       "elapsed_s": time.monotonic() - began, "updated_utc": now()})


if __name__ == "__main__":
    main()
