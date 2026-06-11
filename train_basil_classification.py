#!/usr/bin/env python3
"""
Train YOLOv8 nano CLASSIFICATION on basil_classification_dataset.
Optimizat pentru Apple M1 Pro (10 cores, 16GB unified memory, MPS).

3 clase:
  0: healthy (sanatoasa)
  1: Fusarium oxysporum (fusarioza)
  2: Oidium belbahrii (fainare / powdery mildew)
"""

import sys
import os
from pathlib import Path
import multiprocessing

try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--break-system-packages", "-q"])
    from ultralytics import YOLO

import torch

DATASET = Path("/Users/iustiniuliandodita/sera-hidroponica/basil_classification_dataset")

if not DATASET.exists():
    print(f"Dataset not found: {DATASET}")
    sys.exit(1)

# --- Device ---
if torch.cuda.is_available():
    device = "0"
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
elif torch.backends.mps.is_available():
    device = "mps"
    device_name = "Apple Silicon MPS (M1 Pro)"
else:
    device = "cpu"
    device_name = "CPU"

# --- M1 Pro: max CPU threads for data preprocessing ---
cpu_cores = multiprocessing.cpu_count()          # 10
perf_cores = 8                                   # M1 Pro: 8 performance cores
torch.set_num_threads(perf_cores)
os.environ["OMP_NUM_THREADS"] = str(perf_cores)
os.environ["MKL_NUM_THREADS"] = str(perf_cores)

# Workers: 8 parallel data loader workers (one per performance core)
# Cache "ram": incarca tot datasetul in RAM la start (~23MB - trivial pe 16GB)
# Batch 128: imgsz=224 → ~19MB VRAM per batch, sigur pe unified memory
NUM_WORKERS = perf_cores       # 8
BATCH_SIZE  = 128
CACHE       = "ram"

print("=" * 60)
print("BASIL CLASSIFICATION - YOLOv8 small-cls (v4)")
print("=" * 60)
print(f"Device  : {device_name}")
print(f"Workers : {NUM_WORKERS} (parallel data loaders)")
print(f"Batch   : {BATCH_SIZE}")
print(f"Cache   : {CACHE} (dataset preloaded in RAM)")
print(f"Classes : healthy / Fusarium oxysporum / Oidium belbahrii")
print(f"Train   : 941  |  Val: 269  |  Test: 134")
print()

model = YOLO("yolov8s-cls.pt")

results = model.train(
    data=str(DATASET),
    epochs=200,
    imgsz=416,            # 416: mai aproape de rezolutia originala 392px
    batch=32,
    workers=NUM_WORKERS,
    cache=CACHE,
    device=device,
    patience=15,          # mai multa rabdare — convergenta lenta dar mai buna
    optimizer="AdamW",
    lr0=0.0001,           # mai mic: convergenta ultra-stabila
    lrf=0.001,            # cosine coboara pana la lr0*lrf = 1e-7
    weight_decay=0.002,   # regularizare L2 mai agresiva
    warmup_epochs=5,      # warmup mai lung pentru stabilitate initiala
    cos_lr=True,
    label_smoothing=0.1,
    dropout=0.3,          # dropout mai mare: 0.3 vs 0.2
    amp=True,
    augment=True,
    project="runs/classify",
    name="basil_health_v4",
    exist_ok=False,
)

print("\nTraining complete.")
print("Results : runs/classify/basil_health_v4/")
print("Weights : runs/classify/basil_health_v4/weights/best.pt")
