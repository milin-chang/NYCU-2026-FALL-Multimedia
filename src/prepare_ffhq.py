"""Create a reproducible FFHQ train/validation/test manifest without copying PNGs."""

import argparse
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "dataset/ffhq_1024")
    parser.add_argument("--source-ids", type=Path,
                        default=PROJECT_ROOT / "dataset/ffhq_train_seed42_4000_ids.txt")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "dataset/ffhq_split_seed42_500.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--val", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    args = parser.parse_args()

    counts = {"train": args.train, "val": args.val, "test": args.test}
    if min(counts.values()) <= 0:
        parser.error("All split sizes must be positive.")
    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")
    if not args.source_ids.is_file():
        parser.error(f"Original 4,000-ID list does not exist: {args.source_ids}")

    ids = [line.strip() for line in args.source_ids.read_text(encoding="utf-8").splitlines()]
    if len(ids) != len(set(ids)) or any(not (len(value) == 5 and value.isdigit()) for value in ids):
        parser.error("The original ID list has duplicate or malformed IDs.")
    names = sorted(f"{value}.png" for value in ids)
    actual = sorted(path.name for path in args.images.glob("*.png"))
    if actual != names:
        parser.error("PNG filenames do not match the original 4,000-ID list.")
    if sum(counts.values()) > len(names):
        parser.error(f"Requested {sum(counts.values())} images, but only {len(names)} are available.")

    random.Random(args.seed).shuffle(names)
    chosen = names[:sum(counts.values())]
    splits = {}
    offset = 0
    for split, count in counts.items():
        splits[split] = sorted(chosen[offset:offset + count])
        offset += count

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "selection": "Python random.Random(seed).shuffle(sorted(original 4000 IDs))",
        "source_ids_sha256": hashlib.sha256(args.source_ids.read_bytes()).hexdigest(),
        "counts": counts,
        "splits": splits,
    }
    contents = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        if args.out.read_text(encoding="utf-8") != contents:
            parser.error(f"A different manifest already exists: {args.out}")
    else:
        temporary = args.out.with_suffix(args.out.suffix + ".part")
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(args.out)
    print(f"{args.out}  train={args.train} val={args.val} test={args.test}")
    print(f"manifest SHA-256: {hashlib.sha256(contents.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
