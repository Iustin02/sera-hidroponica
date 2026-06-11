#!/usr/bin/env python3
"""
Train YOLOv8 nano (detection) on basil_detection_3class dataset.
3 clase:
  0: healthy (sanatoasa)
  1: Fusarium oxysporum (fusarioza)
  2: Oidium belbahrii (fainare)
"""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--break-system-packages", "-q"])
    from ultralytics import YOLO

DATASET = Path("/Users/iustiniuliandodita/sera-hidroponica/basil_detection_3class/data.yaml")

if not DATASET.exists():
    print(f"Dataset not found: {DATASET}")
    print("Run build_detection_dataset.py first.")
    sys.exit(1)

print("=" * 60)
print("BASIL DETECTION - 3 CLASS - YOLOv8 nano")
print("=" * 60)
print(f"Dataset : {DATASET}")
print(f"Model   : yolov8n.pt  (detection, not -cls)")
print(f"Classes : healthy / Fusarium oxysporum / Oidium belbahrii")
print()

import torch

if torch.cuda.is_available():
    device = "0"
    print("Device  : CUDA GPU")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Device  : Apple Silicon (MPS)")
else:
    device = "cpu"
    print("Device  : CPU")

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(DATASET),
    epochs=100,
    imgsz=416,
    batch=16,
    device=device,
    patience=5,
    project="runs/detect",
    name="basil_3class_v2",
    mosaic=0.0,        # dezactivat: cu box-uri full-image mosaic scade precision
    box=3.0,           # redus de la 7.5: localizarea e mai putin importanta
    cls=1.5,           # crescut de la 0.5: accent pe clasificare corecta
    exist_ok=False,
)

print("\nTraining complete.")
print(f"Results: runs/detect/basil_3class/")
print(f"Best weights: runs/detect/basil_3class/weights/best.pt")
