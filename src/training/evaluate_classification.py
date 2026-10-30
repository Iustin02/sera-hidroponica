#!/usr/bin/env python3
"""
Evaluare clasificator pe setul de test + matrice de confuzie normalizata (teza, Anexa D).

  --subset roboflow  : doar imaginile Roboflow (1344 total / 134 test) — setul
                       pe care au fost raportate C1-C4 in teza
  --subset all       : dataset-ul extins cu cadre HQ (1736 total / 174 test) — C5

Utilizare:
  python src/training/evaluate_classification.py --weights models/basil_health_c3.pt --imgsz 320 --subset roboflow
  python src/training/evaluate_classification.py --weights models/basil_health_c5.pt --imgsz 416 --subset all
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "basil_classification"
HQ_PREFIX = "hq_frame_"  # cadre extrase din video-ul propriu din sera (adaugate post-teza)


def roboflow_view(tmp: Path) -> Path:
    """Construieste o copie din symlink-uri a dataset-ului fara cadrele HQ."""
    for img in DATASET.glob("*/*/*"):
        if img.name.startswith(HQ_PREFIX) or not img.is_file():
            continue
        dst = tmp / img.relative_to(DATASET)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(img)
    return tmp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(ROOT / "models" / "basil_health_c3.pt"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--subset", default="roboflow", choices=["roboflow", "all"])
    parser.add_argument("--device", default="cpu", help="cpu pentru reproductibilitate pe orice sistem")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    with tempfile.TemporaryDirectory() as tmp:
        data = roboflow_view(Path(tmp)) if args.subset == "roboflow" else DATASET
        metrics = model.val(data=str(data), split=args.split, imgsz=args.imgsz,
                            batch=32, device=args.device, plots=False, verbose=False)

    # Matricea Ultralytics: linii = prezis, coloane = real -> transpunem
    names = [model.names[i] for i in sorted(model.names)]
    cm = metrics.confusion_matrix.matrix[: len(names), : len(names)].T
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    print(f"\nModel   : {args.weights}")
    print(f"Subset  : {args.subset} / {args.split} ({int(cm.sum())} imagini)")
    print(f"Top-1   : {metrics.top1:.2%}   Top-5: {metrics.top5:.2%}\n")
    print("Matrice de confuzie normalizata (linii = reale, coloane = prezise):")
    print(" " * 12 + "".join(f"{n:>10s}" for n in names))
    for name, row in zip(names, cm_norm):
        print(f"{name:>12s}" + "".join(f"{v:10.2f}" for v in row))
    print("\nRecall per clasa:")
    for name, acc in zip(names, np.diag(cm_norm)):
        print(f"  {name:10s} {acc:.1%}")


if __name__ == "__main__":
    main()
