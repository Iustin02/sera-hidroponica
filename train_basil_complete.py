#!/usr/bin/env python3
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--break-system-packages", "-q"])
    from ultralytics import YOLO

print("=" * 60)
print("🌿 YOLOv8 BASIL DISEASE DETECTION - TRAINING")
print("=" * 60)

dataset_config = Path("basil_dataset/data.yaml")
if not dataset_config.exists():
    print(f"❌ {dataset_config} not found!")
    sys.exit(1)

print("\n📦 Loading YOLOv8 nano...")
model = YOLO("yolov8n.pt")

print("\n🚀 Starting training...")
results = model.train(
    data=str(dataset_config),
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu",
    patience=10,
    save=True,
    verbose=True
)

print("\n✅ Training complete!")
print("📊 Results saved to: runs/detect/train/")

