#!/usr/bin/env python3
"""Auto-label bounding boxes for herbs (basil/mint) from iNaturalist dataset.

Goal: replace the full-frame bbox with a tighter bbox around the plant area.
Heuristic: HSV green mask -> largest connected component -> padded bounding box.

Outputs a "run" folder under runs\autolabel\<timestamp> containing:
- images/: copied source images
- labels/: YOLO txt labels (same class id, new bbox)
- preview/: images with bbox drawn
- manifest.csv: per-image info + fallback flags

This is meant as a quick bootstrap; review previews before scaling to full dataset.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    split: str  # train/val
    img_path: Path
    lbl_path: Path


def _read_class_id(lbl_path: Path) -> int:
    txt = lbl_path.read_text(encoding="utf-8").strip().split()
    if not txt:
        raise ValueError(f"Empty label file: {lbl_path}")
    return int(float(txt[0]))


def _iter_samples(dataset_root: Path, train_rel: str, val_rel: str) -> list[Sample]:
    out: list[Sample] = []

    for split, rel in [("train", train_rel), ("val", val_rel)]:
        img_dir = (dataset_root / rel).resolve()
        if not img_dir.exists():
            continue

        # Map images -> labels by mirroring path segment images/<split> -> labels/<split>
        # Example: images/train/abc.jpg -> labels/train/abc.txt
        for img in img_dir.rglob("*"):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue

            try:
                parts = list(img.parts)
                # Replace the first occurrence of "images" with "labels"
                if "images" in parts:
                    i = parts.index("images")
                    parts[i] = "labels"
                    lbl = Path(*parts).with_suffix(".txt")
                else:
                    # Fallback: labels under dataset_root/labels/<split>/<stem>.txt
                    lbl = dataset_root / "labels" / split / f"{img.stem}.txt"

                if lbl.exists():
                    out.append(Sample(split=split, img_path=img, lbl_path=lbl))
            except Exception:
                continue

    return out


def _green_bbox(img_bgr, min_area_ratio: float = 0.01):
    """Return (x1,y1,x2,y2, ok, mask_area_ratio)."""
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Heuristic green range (broad)
    lower = (25, 35, 35)
    upper = (95, 255, 255)
    mask = cv2.inRange(img_hsv, lower, upper)

    # Morphology to remove noise
    k = max(3, int(round(min(h, w) * 0.01)) | 1)  # odd kernel size ~1% of min dim
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    area = int(cv2.countNonZero(mask))
    area_ratio = area / float(h * w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w - 1, h - 1, False, area_ratio

    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)

    ok = area_ratio >= min_area_ratio and bw > 5 and bh > 5

    # Pad bbox by 5%
    pad_x = int(round(bw * 0.05))
    pad_y = int(round(bh * 0.05))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w - 1, x + bw + pad_x)
    y2 = min(h - 1, y + bh + pad_y)

    return x1, y1, x2, y2, ok, area_ratio


def _xyxy_to_yolo(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    # Clamp
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return cx, cy, bw, bh


def main():
    ap = argparse.ArgumentParser(description="Auto-label tighter YOLO boxes for herbs (basil/mint)")
    ap.add_argument(
        "--data",
        type=str,
        default=str(Path("inat_datasets/inat_basil_mint_20k/data.yaml")),
        help="Path to a YOLO data.yaml (defaults to inat_datasets/inat_basil_mint_20k/data.yaml)",
    )
    ap.add_argument("--n", type=int, default=100, help="Number of random images to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output run directory (default: runs/autolabel/<timestamp>)",
    )
    ap.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.01,
        help="Minimum green-mask area ratio to consider bbox 'ok' (otherwise fallback is used)",
    )

    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found: {data_yaml}")
        return 2

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    train_rel = cfg.get("train", "images/train")
    val_rel = cfg.get("val", "images/val")

    all_samples = _iter_samples(root, train_rel, val_rel)
    if not all_samples:
        print(f"ERROR: no samples found under dataset root: {root}")
        return 2

    random.seed(args.seed)
    random.shuffle(all_samples)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) if args.out else (Path("runs") / "autolabel" / ts)

    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    preview_out = out_dir / "preview"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    preview_out.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"

    chosen: list[Sample] = []
    for s in all_samples:
        if len(chosen) >= args.n:
            break
        chosen.append(s)

    n_ok = 0
    n_fallback = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "split",
            "src_image",
            "src_label",
            "dst_image",
            "dst_label",
            "class_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "mask_area_ratio",
            "bbox_ok",
        ])

        for idx, s in enumerate(chosen, start=1):
            img = cv2.imread(str(s.img_path))
            if img is None:
                continue

            h, wimg = img.shape[:2]
            class_id = _read_class_id(s.lbl_path)

            x1, y1, x2, y2, ok, area_ratio = _green_bbox(img, min_area_ratio=args.min_area_ratio)

            if not ok:
                # Fallback: use a centered crop bbox (80% of image)
                bw = int(round(wimg * 0.8))
                bh = int(round(h * 0.8))
                x1 = max(0, (wimg - bw) // 2)
                y1 = max(0, (h - bh) // 2)
                x2 = min(wimg - 1, x1 + bw)
                y2 = min(h - 1, y1 + bh)
                n_fallback += 1
            else:
                n_ok += 1

            cx, cy, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, wimg, h)

            dst_img = images_out / f"{idx:04d}_{s.img_path.name}"
            dst_lbl = labels_out / f"{dst_img.stem}.txt"

            # Copy image
            dst_img.write_bytes(s.img_path.read_bytes())

            # Write new label (single bbox)
            dst_lbl.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

            # Preview
            preview = img.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(
                preview,
                f"cls={class_id} ok={int(ok)}",
                (max(5, x1), max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(preview_out / f"{dst_img.stem}.jpg"), preview)

            w.writerow([
                s.split,
                str(s.img_path),
                str(s.lbl_path),
                str(dst_img),
                str(dst_lbl),
                class_id,
                x1,
                y1,
                x2,
                y2,
                f"{area_ratio:.6f}",
                int(ok),
            ])

    print("=" * 60)
    print("AUTO-LABEL RUN COMPLETE")
    print("=" * 60)
    print(f"Output: {out_dir}")
    print(f"Images: {len(list(images_out.glob('*')))}")
    print(f"Labels: {len(list(labels_out.glob('*.txt')))}")
    print(f"Preview: {len(list(preview_out.glob('*.jpg')))}")
    print(f"bbox_ok: {n_ok} | fallback: {n_fallback}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
