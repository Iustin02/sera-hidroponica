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
import numpy as np

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


def _hsv_green_mask(img_bgr, *, strict: bool) -> "cv2.Mat":
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Broad vs strict green range.
    if strict:
        # stricter -> less background vegetation
        lower = (30, 80, 40)
        upper = (90, 255, 255)
    else:
        lower = (25, 35, 35)
        upper = (95, 255, 255)

    mask = cv2.inRange(img_hsv, lower, upper)

    # Morphology to remove noise
    k = max(3, int(round(min(h, w) * 0.01)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _best_component_bbox(mask: "cv2.Mat"):
    """Return (x,y,bw,bh, area_ratio, touches_border) for best component or None."""
    h, w = mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    best = None
    best_score = -1.0

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if bw <= 5 or bh <= 5:
            continue

        touches = x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h
        area_ratio = area / float(h * w)

        # Penalize border-touching components (often background vegetation).
        score = area * (0.25 if touches else 1.0)
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh, area_ratio, touches)

    return best


def _bbox_from_mask(mask: "cv2.Mat", min_area_ratio: float):
    h, w = mask.shape[:2]
    area = int(cv2.countNonZero(mask))
    area_ratio = area / float(h * w)

    comp = _best_component_bbox(mask)
    if comp is None:
        return 0, 0, w - 1, h - 1, False, area_ratio, 1.0, True

    x, y, bw, bh, _comp_area_ratio, touches = comp

    ok = area_ratio >= min_area_ratio and bw > 5 and bh > 5

    # Pad bbox by 4%
    pad_x = int(round(bw * 0.04))
    pad_y = int(round(bh * 0.04))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w - 1, x + bw + pad_x)
    y2 = min(h - 1, y + bh + pad_y)

    bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
    return x1, y1, x2, y2, ok, area_ratio, bbox_area_ratio, touches


def _apply_edge_margin(x1, y1, x2, y2, w, h, margin_ratio: float = 0.01):
    """Ensure bbox isn't exactly full-frame by enforcing a small inner margin."""
    mx = int(round(w * margin_ratio))
    my = int(round(h * margin_ratio))
    x1 = max(x1, mx)
    y1 = max(y1, my)
    x2 = min(x2, w - 1 - mx)
    y2 = min(y2, h - 1 - my)
    if x2 <= x1:
        x1, x2 = 0, w - 1
    if y2 <= y1:
        y1, y2 = 0, h - 1
    return x1, y1, x2, y2


def _grabcut_bbox(img_bgr, fg_mask: "cv2.Mat", min_area_ratio: float):
    """Use GrabCut seeded by fg_mask + border as background. Returns bbox tuple like _bbox_from_mask."""
    h, w = img_bgr.shape[:2]

    gc = cv2.GC_PR_BGD * (np.ones((h, w), dtype=np.uint8))

    # Sure background on a border strip
    m = max(5, int(round(min(h, w) * 0.03)))
    gc[:m, :] = cv2.GC_BGD
    gc[-m:, :] = cv2.GC_BGD
    gc[:, :m] = cv2.GC_BGD
    gc[:, -m:] = cv2.GC_BGD

    # Probable foreground from provided mask
    gc[fg_mask > 0] = cv2.GC_PR_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, gc, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return 0, 0, w - 1, h - 1, False, 0.0, 1.0, True

    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    return _bbox_from_mask(out, min_area_ratio=min_area_ratio)


def _plant_bbox(img_bgr, min_area_ratio: float = 0.01):
    """Return (x1,y1,x2,y2, ok, mask_area_ratio, bbox_area_ratio, method)."""
    import numpy as np  # local import to keep global deps minimal

    h, w = img_bgr.shape[:2]

    broad = _hsv_green_mask(img_bgr, strict=False)
    strict = _hsv_green_mask(img_bgr, strict=True)

    # Try strict first (usually tighter)
    x1, y1, x2, y2, ok, area_ratio, bbox_ratio, touches = _bbox_from_mask(strict, min_area_ratio)
    method = "hsv_strict"

    # If strict is too small, fallback to broad
    if not ok:
        x1, y1, x2, y2, ok, area_ratio, bbox_ratio, touches = _bbox_from_mask(broad, min_area_ratio)
        method = "hsv_broad"

    # If bbox is huge or border-touching, try removing border-connected green then re-bbox
    if ok and (bbox_ratio > 0.92 or touches):
        # Remove components touching border by zeroing them
        def strip_border(mask):
            hh, ww = mask.shape[:2]
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            keep = np.zeros_like(mask)
            for i in range(1, n):
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                touches_b = x <= 0 or y <= 0 or (x + bw) >= ww or (y + bh) >= hh
                if not touches_b:
                    keep[labels == i] = 255
            return keep

        stripped = strip_border(strict)
        x1b, y1b, x2b, y2b, okb, area_r_b, bbox_r_b, touches_b = _bbox_from_mask(stripped, min_area_ratio)
        if okb and bbox_r_b < bbox_ratio:
            x1, y1, x2, y2, ok, area_ratio, bbox_ratio, touches = x1b, y1b, x2b, y2b, okb, area_r_b, bbox_r_b, touches_b
            method = "hsv_strict_stripborder"

    # If still huge, run GrabCut using strict mask as seed
    if ok and bbox_ratio > 0.95:
        x1g, y1g, x2g, y2g, okg, area_r_g, bbox_r_g, touches_g = _grabcut_bbox(img_bgr, strict, min_area_ratio)
        if okg and bbox_r_g < bbox_ratio:
            x1, y1, x2, y2, ok, area_ratio, bbox_ratio, touches = x1g, y1g, x2g, y2g, okg, area_r_g, bbox_r_g, touches_g
            method = "grabcut"

    # Avoid exact full-frame boxes
    x1, y1, x2, y2 = _apply_edge_margin(x1, y1, x2, y2, w, h, margin_ratio=0.01)
    bbox_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)

    return x1, y1, x2, y2, ok, area_ratio, bbox_ratio, method


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
            "bbox_area_ratio",
            "bbox_ok",
            "method",
        ])

        for idx, s in enumerate(chosen, start=1):
            img = cv2.imread(str(s.img_path))
            if img is None:
                continue

            h, wimg = img.shape[:2]
            class_id = _read_class_id(s.lbl_path)

            x1, y1, x2, y2, ok, area_ratio, bbox_ratio, method = _plant_bbox(img, min_area_ratio=args.min_area_ratio)

            if not ok:
                # Fallback: use a centered crop bbox (80% of image)
                bw = int(round(wimg * 0.8))
                bh = int(round(h * 0.8))
                x1 = max(0, (wimg - bw) // 2)
                y1 = max(0, (h - bh) // 2)
                x2 = min(wimg - 1, x1 + bw)
                y2 = min(h - 1, y1 + bh)
                bbox_ratio = ((x2 - x1) * (y2 - y1)) / float(wimg * h)
                method = "fallback_center"
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
                f"cls={class_id} ok={int(ok)} {method} box={bbox_ratio:.2f}",
                (max(5, x1), max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
                f"{bbox_ratio:.6f}",
                int(ok),
                method,
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
