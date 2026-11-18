"""Freshly rerun all eight metrics on 50 FFHQ test faces for three best checkpoints."""

import csv
import hashlib
import json
from pathlib import Path

import torch

from benchmark_swinir import make_model
from quality_eval import METRIC_DIRECTIONS, MetricSuite
from train_lightweight import evaluate, now, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "three_runs_convergence" / "test_reevaluation"
RUNS = (
    ("bicubic", "swinir_s_ffhq500_x4_seed42_150k_20261118"),
    ("highfreq", "swinir_s_ffhq500_x4_highfreq_s8_seed42_100k_20261118"),
    ("masked", "swinir_s_ffhq500_x4_masked_s16_seed42_100k_20261118"),
)
METRICS = tuple(METRIC_DIRECTIONS)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_rows(previous, fresh):
    with previous.open(encoding="utf-8", newline="") as stream:
        old_rows = list(csv.DictReader(stream))
    with fresh.open(encoding="utf-8", newline="") as stream:
        new_rows = list(csv.DictReader(stream))
    if len(old_rows) != 50 or len(new_rows) != 50:
        raise ValueError("Expected exactly 50 test scores in each CSV.")
    if [row["image"] for row in old_rows] != [row["image"] for row in new_rows]:
        raise ValueError("The previous and new test image lists differ.")
    return {metric: max(abs(float(old[metric]) - float(new[metric]))
                        for old, new in zip(old_rows, new_rows))
            for metric in METRICS}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the original BF16 test inference.")
    device = torch.device("cuda")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support the original BF16 inference precision.")
    suite = MetricSuite(device)
    results = []
    for key, name in RUNS:
        spec = json.loads((ROOT / "spec" / f"{name}.json").read_text(encoding="utf-8"))
        manifest = json.loads(Path(spec["data"]["manifest"]).read_text(encoding="utf-8"))
        names = manifest["splits"]["test"]
        if len(names) != 50 or spec["train"]["test_limit"] is not None:
            raise ValueError(f"Expected a full 50-image test set for {name}.")
        checkpoint = ROOT / "runs" / name / "checkpoints" / "best.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = make_model().eval().to(device)
        model.load_state_dict(state["ema"])
        destination = OUTPUT / key
        scores = evaluate(model, suite, names, Path(spec["data"]["images"]),
                          Path(spec["data"]["lr_images"]), device, "bf16", destination)
        del model, state
        torch.cuda.empty_cache()
        differences = compare_rows(ROOT / "runs" / name / "test" / "per_image.csv",
                                   destination / "per_image.csv")
        results.append({"key": key, "run": name, "test_images": len(names),
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "metrics": scores, "maximum_per_image_difference_from_original": differences})
        print(f"Reevaluated {name}: PSNR={scores['psnr']:.4f} dB; "
              f"maximum metric difference={max(differences.values()):.8g}", flush=True)
    write_json(OUTPUT / "results.json", {
        "completed_utc": now(), "split": "FFHQ-500 seed 42 test", "images_per_run": 50,
        "inference_weights": "EMA state in each best.pth", "precision": "CUDA BF16",
        "metric_directions": METRIC_DIRECTIONS, "runs": results,
    })
    print(f"Saved fresh test reevaluation to {OUTPUT}")


if __name__ == "__main__":
    main()
