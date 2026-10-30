#!/usr/bin/env python3
"""
Antrenare detector YOLOv8 (bounding boxes) pe data/basil_detection_3class.

Ramura experimentala (teza, Cap. 5.4). Etichetele sunt bbox-uri generice
centrate (0.5 0.5 0.85 0.85), deci modelul invata clasa, nu localizarea leziunii.

Presetari:
  D1 — configuratie implicita Ultralytics (mosaic activ)            -> basil_3class
  D2 — mosaic dezactivat, accent pe clasificare (box=3.0, cls=1.5)  -> basil_3class_v2

Utilizare:
  python src/training/train_detection.py --preset D1
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "data" / "basil_detection_3class" / "data.yaml"
PROJECT = ROOT / "experiments" / "detection"

PRESETS = {
    "D1": dict(run="basil_3class"),
    "D2": dict(run="basil_3class_v2", mosaic=0.0, box=3.0, cls=1.5),
}


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="D1")
    parser.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt | yolov8s.pt")
    parser.add_argument("--name")
    parser.add_argument("--device")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        sys.exit(f"Lipseste {DATA_YAML}. Ruleaza intai src/data/build_detection_dataset.py")

    from ultralytics import YOLO

    cfg = dict(PRESETS[args.preset])
    run_name = args.name or cfg.pop("run")
    cfg.pop("run", None)

    YOLO(args.model).train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=416,
        batch=16,
        patience=5,
        device=args.device or select_device(),
        seed=0,
        deterministic=True,
        project=str(PROJECT),
        name=run_name,
        exist_ok=False,
        **cfg,
    )
    print(f"Rezultate: {PROJECT / run_name}")


if __name__ == "__main__":
    main()
