#!/usr/bin/env python3
"""
Antrenare clasificator YOLOv8-cls pentru starea fitosanitara a busuiocului.

Reproduce experimentele C1-C5 din teza (Cap. 4.3.4, Anexa C). Fiecare preset
contine exact hiperparametrii salvati in experiments/classification/<run>/args.yaml.

Clase (ordinea interna Ultralytics = alfabetica dupa numele folderului):
  0: fusarium  -> Fusarium oxysporum (fuzarioza)
  1: healthy   -> planta sanatoasa
  2: powdery   -> Oidium belbahrii (fainare)

Utilizare:
  python src/training/train_classification.py --preset C3
  python src/training/train_classification.py --preset C5 --name basil_health_v6
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "basil_classification"
PROJECT = ROOT / "experiments" / "classification"

# Hiperparametri comuni C2-C5 (regularizare introdusa in C2)
_REGULARIZED = dict(batch=32, optimizer="AdamW", cos_lr=True, label_smoothing=0.1)

PRESETS = {
    # C1 — referinta: nano, imgsz mic, batch mare (overfitting sever)
    "C1": dict(model="yolov8n-cls.pt", run="basil_health_v1", epochs=100, patience=5,
               batch=128, imgsz=224, optimizer="AdamW", lr0=0.001, lrf=0.01,
               weight_decay=0.0005, warmup_epochs=3, cos_lr=False, dropout=0.0),
    # C2 — batch 32, imgsz 320, dropout + label smoothing + cosine LR
    "C2": dict(model="yolov8n-cls.pt", run="basil_health_v2", epochs=150, patience=10,
               imgsz=320, lr0=0.0003, lrf=0.01, weight_decay=0.001, warmup_epochs=3,
               dropout=0.2, **_REGULARIZED),
    # C3 — identic cu C2, model small (modelul raportat in teza: 81,04% Top-1)
    "C3": dict(model="yolov8s-cls.pt", run="basil_health_v3", epochs=150, patience=10,
               imgsz=320, lr0=0.0003, lrf=0.01, weight_decay=0.001, warmup_epochs=3,
               dropout=0.2, **_REGULARIZED),
    # C4 — imgsz 416, LR mai mic, regularizare mai puternica
    "C4": dict(model="yolov8s-cls.pt", run="basil_health_v4", epochs=200, patience=15,
               imgsz=416, lr0=0.0001, lrf=0.001, weight_decay=0.002, warmup_epochs=5,
               dropout=0.3, **_REGULARIZED),
    # C5 — configuratia C4 pe dataset-ul extins cu cadre HQ (post-teza, 86,74% Top-1)
    "C5": dict(model="yolov8s-cls.pt", run="basil_health_v4-2", epochs=200, patience=15,
               imgsz=416, lr0=0.0001, lrf=0.001, weight_decay=0.002, warmup_epochs=5,
               dropout=0.3, **_REGULARIZED),
}


def select_device() -> str:
    """Detectie automata dispozitiv (CUDA > MPS > CPU)."""
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="C5")
    parser.add_argument("--name", help="nume run (implicit: numele din preset + sufix automat)")
    parser.add_argument("--epochs", type=int, help="suprascrie numarul maxim de epoci")
    parser.add_argument("--device", help="cpu | mps | 0 (implicit: autodetectie)")
    parser.add_argument("--workers", type=int, default=min(8, multiprocessing.cpu_count()))
    args = parser.parse_args()

    if not DATASET.exists():
        sys.exit(f"Dataset inexistent: {DATASET}")

    from ultralytics import YOLO

    cfg = dict(PRESETS[args.preset])
    model_name = cfg.pop("model")
    run_name = args.name or cfg.pop("run")
    cfg.pop("run", None)
    if args.epochs:
        cfg["epochs"] = args.epochs

    # Fire CPU pentru preprocesare (M1 Pro: 8 performance cores)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.workers))

    device = args.device or select_device()
    print(f"Preset {args.preset} | model {model_name} | device {device} | run {run_name}")

    YOLO(model_name).train(
        data=str(DATASET),
        device=device,
        workers=args.workers,
        cache="ram",       # preincarca dataset-ul in RAM (~23 MB)
        amp=True,          # Automatic Mixed Precision
        augment=True,
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
