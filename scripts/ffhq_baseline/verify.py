"""Verify the prepared FFHQ split and compare it with the existing split."""

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
PREPARED = DATA / "ffhq_baseline_1000_125_125"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    manifest_path = PREPARED / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Source: {manifest['source']}")
    print(f"Candidates: {manifest['candidate_count']}; seed: {manifest['seed']}")
    all_names = set()
    for split, rows in manifest["splits"].items():
        names = {row["name"] for row in rows}
        assert len(names) == len(rows) and not (all_names & names)
        all_names.update(names)
        assert len(list((PREPARED / split / "hr").glob("*.png"))) == len(rows)
        if split != "train":
            assert len(list((PREPARED / split / "lr").glob("*.png"))) == len(rows)
        for row in rows:
            hr_path = PREPARED / split / "hr" / row["name"]
            assert sha256(hr_path) == row["hr_sha256"]
            with Image.open(hr_path) as image:
                assert image.mode == "RGB" and image.size == (512, 512)
            if split != "train":
                lr_path = PREPARED / split / "lr" / row["name"]
                assert sha256(lr_path) == row["lr_sha256"]
                with Image.open(lr_path) as image:
                    assert image.mode == "RGB" and image.size == (128, 128)
        print(f"{split}: {len(rows)} verified")

    old_split = json.loads((DATA / "ffhq_split_seed42_500.json").read_text(encoding="utf-8"))["splits"]
    old_pool = {f"{name}.png" for name in (DATA / "ffhq_train_seed42_4000_ids.txt").read_text().splitlines()}
    for split, rows in manifest["splits"].items():
        names = {row["name"] for row in rows}
        overlaps = {"old_4000": len(names & old_pool)}
        overlaps.update({f"old_{part}": len(names & set(values)) for part, values in old_split.items()})
        print(f"{split} overlap: {overlaps}")
    print(f"Manifest SHA-256: {sha256(manifest_path)}")


if __name__ == "__main__":
    main()
