#!/usr/bin/env python3
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--break-system-packages", "-q"])
    from ultralytics import YOLO

root = Path("/Users/iustiniuliandodita/sera-hidroponica/basil_classification_dataset")
if not root.exists():
    print(f"❌ Dataset not found: {root}")
    sys.exit(1)

print("=" * 60)
print("🌿 BASIL CLASSIFICATION TRAINING")
print("=" * 60)
print(f"Dataset: {root}")
print("Model: yolov8n-cls.pt")

model = YOLO("yolov8n-cls.pt")
results = model.train(
    data=str(root),
    epochs=10,
    imgsz=224,
    batch=16,
    device="cpu",
    patience=3,
    project="runs/classify",
    name="basil_health"
)

print("\n✅ Training complete")
print("Results: runs/classify/basil_health/")
