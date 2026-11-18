"""Time lightweight SwinIR-S x4 training steps from random weights on FFHQ."""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAIR_ROOT = Path(__file__).resolve().parent / "upstream/KAIR"
KAIR_COMMIT = "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e"
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
sys.path.insert(0, str(KAIR_ROOT))


def make_model():
    from network_swinir import SwinIR

    return SwinIR(img_size=64, patch_size=1, in_chans=3, embed_dim=60,
                  depths=[6] * 4, num_heads=[6] * 4, window_size=8, mlp_ratio=2,
                  upscale=4, img_range=1.0, upsampler="pixelshuffledirect",
                  resi_connection="1conv")


def make_pair(path, lr_path, seed, return_transform=False):
    from utils.utils_image import augment_img

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    with Image.open(path) as raw:
        hr = np.asarray(ImageOps.exif_transpose(raw).convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(lr_path) as raw:
        lr = np.asarray(raw.convert("RGB"), dtype=np.float32) / 255.0
    if hr.shape[0] != lr.shape[0] * 4 or hr.shape[1] != lr.shape[1] * 4:
        raise ValueError(f"LR/HR dimensions do not match x4: {path}, {lr_path}")
    x = random.randrange(lr.shape[1] - 64 + 1)
    y = random.randrange(lr.shape[0] - 64 + 1)
    lr = lr[y:y + 64, x:x + 64]
    hr = hr[4 * y:4 * y + 256, 4 * x:4 * x + 256]
    mode = random.randint(0, 7)
    lr, hr = augment_img(lr, mode), augment_img(hr, mode)
    pair = (torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1), dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(hr.transpose(2, 0, 1), dtype=np.float32)))
    if return_transform:
        return (*pair, x, y, mode)
    return pair


def load_batch(paths, lr_images, position, batch_size, seed, device):
    pairs = []
    for i in range(batch_size):
        path = paths[(position * batch_size + i) % len(paths)]
        pairs.append(make_pair(path, lr_images / path.name, seed + position * batch_size + i))
    lr = torch.stack([pair[0] for pair in pairs]).to(device)
    hr = torch.stack([pair[1] for pair in pairs]).to(device)
    torch.cuda.synchronize(device)
    return lr, hr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "dataset/ffhq_split_seed42_500.json")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "dataset/ffhq_1024")
    parser.add_argument("--lr-images", type=Path,
                        default=PROJECT_ROOT / "dataset/ffhq_lightweight_x4/lr")
    parser.add_argument("--steps", type=int, default=10, help="Timed optimizer updates")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed optimizer updates")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "src/runs/benchmark_lightweight_500.json")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.batch < 1:
        parser.error("--steps and --batch must be positive; --warmup cannot be negative.")
    if not KAIR_ROOT.is_dir():
        parser.error("Missing src/upstream/KAIR. Run the setup instructions in src/README.md.")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment; install CUDA PyTorch first.")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("This GPU/PyTorch build does not support BF16; use --precision fp32.")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = [args.images / name for name in manifest["splits"]["train"]]
    if not paths or any(not path.is_file() or not (args.lr_images / path.name).is_file()
                        for path in paths):
        parser.error("Training HR/LR images are missing. Run src/prepare_lightweight_lr.py first.")
    seed = manifest["seed"]
    random.Random(seed).shuffle(paths)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = make_model().train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    use_amp = args.precision == "bf16"
    torch.cuda.reset_peak_memory_stats(device)
    measurements = []
    print(f"GPU: {torch.cuda.get_device_name(device)} | torch: {torch.__version__}", flush=True)
    print(f"SwinIR-S lightweight x4, batch={args.batch}, precision={args.precision}, "
          f"warmup={args.warmup}, timed={args.steps}", flush=True)
    for position in range(args.warmup + args.steps):
        started = time.perf_counter()
        lr, hr = load_batch(paths, args.lr_images, position, args.batch, seed, device)
        data_done = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            sr = model(lr)
            loss = F.l1_loss(sr, hr)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        if position >= args.warmup:
            item = {"step": position - args.warmup + 1,
                    "data_s": data_done - started,
                    "gpu_update_s": finished - data_done,
                    "total_s": finished - started,
                    "loss": loss.item()}
            measurements.append(item)
            print(f"step {item['step']:>3}: total={item['total_s']:.3f}s "
                  f"(data={item['data_s']:.3f}s, update={item['gpu_update_s']:.3f}s) "
                  f"loss={item['loss']:.4f}", flush=True)
    totals = [item["total_s"] for item in measurements]
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "kair_commit": KAIR_COMMIT,
        "variant": "lightweight_x4",
        "degradation": "precomputed KAIR imresize_np bicubic x4, antialiasing, 8-bit PNG",
        "upsampler": "pixelshuffledirect",
        "manifest": str(args.manifest),
        "manifest_seed": seed,
        "train_images": len(paths),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "lr_shape": list(lr.shape),
        "hr_shape": list(hr.shape),
        "sr_shape": list(sr.shape),
        "batch": args.batch,
        "precision": args.precision,
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "mean_total_s": statistics.mean(totals),
        "median_total_s": statistics.median(totals),
        "mean_data_s": statistics.mean(item["data_s"] for item in measurements),
        "mean_gpu_update_s": statistics.mean(item["gpu_update_s"] for item in measurements),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        "steps": measurements,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Mean: {report['mean_total_s']:.3f}s/step; "
          f"GPU update: {report['mean_gpu_update_s']:.3f}s/step; "
          f"peak allocated: {report['peak_allocated_gib']:.2f} GiB", flush=True)
    print(f"Report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
