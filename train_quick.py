#!/usr/bin/env python3
"""
Quick training test with sample data (50 images)
Uses CPU since GPU is not available
"""

import sys
from pathlib import Path

print("=" * 60)
print("🚀 BASIL YOLOv8 - QUICK TEST TRAINING")
print("=" * 60)

# Check dataset
dataset_config = Path("basil_dataset/data.yaml")
if not dataset_config.exists():
    print(f"❌ {dataset_config} not found!")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("📥 Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--break-system-packages", "-q"])
    from ultralytics import YOLO

print("\n📊 Dataset info:")
print(f"   ✅ Config: {dataset_config}")
print(f"   ✅ Images: 50 (30 train, 17 val, 3 test)")

print("\n📦 Loading YOLOv8 nano...")
model = YOLO("yolov8n.pt")

print("\n🎯 Training started...")
print("   Model: YOLOv8 nano")
print("   Epochs: 1 (quick test)")
print("   Batch: 4")
print("   Device: CPU")

results = model.train(
    data=str(dataset_config),
    epochs=1,
    imgsz=416,
    batch=4,
    device="cpu",
    patience=1,
    save=True,
    verbose=False,
    project="runs/detect",
    name="basil_test_quick"
)

print("\n✅ Training complete!")
print(f"\n📊 Results:")
print(f"   Folder: runs/detect/basil_test_quick/")
print(f"   Best model: runs/detect/basil_test_quick/weights/best.pt")
print(f"\n💡 Next: Check results or train with real Roboflow data")
