#!/usr/bin/env python3
"""
Export YOLOv8s-cls -> ONNX pentru deployment pe Raspberry Pi 5 (teza, Anexa H).

ONNX (opset 12) ruleaza cu onnxruntime, fara dependenta de PyTorch la inferenta.

Utilizare:
  python src/edge/export_onnx.py --weights models/basil_health_c3.pt --imgsz 320
  python src/edge/export_onnx.py --weights models/basil_health_c5.pt --imgsz 416
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def export_onnx(weights: Path, imgsz: int, out_dir: Path) -> Path:
    from ultralytics import YOLO

    exported = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=12,
                                              simplify=True, dynamic=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{weights.stem}.onnx"
    exported.replace(target)
    print(f"Exportat: {target} ({target.stat().st_size / 1024:.1f} KB)")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "basil_health_c3.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    export_onnx(args.weights, args.imgsz, args.out)
