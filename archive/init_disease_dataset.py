#!/usr/bin/env python3
"""Initialize a local YOLO dataset skeleton for disease/deficiency detection.

Why:
- iNaturalist basil/mint dataset only has species labels (basil vs mint), not disease labels.
- To train classes like early_blight/late_blight/deficiency/etc you need a dataset whose labels
  use those class IDs.

This script creates:
  <out>/images/{train,val}
  <out>/labels/{train,val}
  <out>/data.yaml

Optionally it can copy a small random subset of images from an existing YOLO dataset
(e.g., iNat) to bootstrap manual labeling.

Note: copying images does NOT create disease labels; you still need to annotate symptoms.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


DEFAULT_CLASSES_4 = [
    "healthy",
    "fungal",
    "bacterial",
    "abiotic_deficiency",
]

DEFAULT_CLASSES_GRANULAR = [
    "healthy",
    "downy_mildew",
    "powdery_mildew",
    "rust",
    "leaf_spot",
    "bacterial_leaf_spot",
    "wilt",
    "chlorosis_deficiency",
    "salt_burn_overfert",
    "pest_damage",
]


def _gather_images(src_images_dir: Path) -> list[Path]:
    imgs = [p for p in src_images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(imgs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Init YOLO dataset skeleton for disease/deficiency classes")
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path("datasets") / "herbs_disease"),
        help="Output dataset root (default: datasets/herbs_disease)",
    )
    ap.add_argument(
        "--schema",
        choices=["4", "granular"],
        default="4",
        help="Class schema: 4 (general) or granular",
    )
    ap.add_argument(
        "--copy-from",
        type=str,
        default=None,
        help="Optional: path to an existing YOLO dataset root (must contain images/train and images/val)",
    )
    ap.add_argument("--n-train", type=int, default=0, help="Optional: number of train images to copy")
    ap.add_argument("--n-val", type=int, default=0, help="Optional: number of val images to copy")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out = Path(args.out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    classes = DEFAULT_CLASSES_4 if args.schema == "4" else DEFAULT_CLASSES_GRANULAR

    data = {
        "path": str(out).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": {i: n for i, n in enumerate(classes)},
    }

    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if args.copy_from and (args.n_train or args.n_val):
        src = Path(args.copy_from)
        src_train = src / "images" / "train"
        src_val = src / "images" / "val"
        if not src_train.exists() or not src_val.exists():
            raise SystemExit(f"copy-from must contain images/train and images/val: {src}")

        random.seed(args.seed)

        if args.n_train:
            imgs = _gather_images(src_train)
            random.shuffle(imgs)
            for p in imgs[: args.n_train]:
                (out / "images" / "train" / p.name).write_bytes(p.read_bytes())

        if args.n_val:
            imgs = _gather_images(src_val)
            random.shuffle(imgs)
            for p in imgs[: args.n_val]:
                (out / "images" / "val" / p.name).write_bytes(p.read_bytes())

    print("Dataset initialized:")
    print(" ", out)
    print("data.yaml:")
    print(" ", out / "data.yaml")
    print("Classes:")
    for i, n in enumerate(classes):
        print(f"  {i}: {n}")

    if args.copy_from and (args.n_train or args.n_val):
        print("\nCopied images (labels still need manual annotation):")
        print(" train:", len(list((out / "images" / "train").glob("*"))))
        print(" val:  ", len(list((out / "images" / "val").glob("*"))))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
