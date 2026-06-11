#!/usr/bin/env python3
"""
Train YOLOv8 model for Basil disease detection
"""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q", "--break-system-packages"])
    from ultralytics import YOLO

def train_basil_model():
    """Train YOLOv8 on Basil dataset"""
    
    dataset_config = "basil_dataset/data.yaml"
    
    if not Path(dataset_config).exists():
        print(f"❌ Error: {dataset_config} not found!")
        print("   Please prepare dataset first")
        return False
    
    print("=" * 60)
    print("🌿 TRAINING YOLOv8 FOR BASIL DISEASE DETECTION")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")
    
    # Train
    print("\n🚀 Starting training...")
    results = model.train(
        data=dataset_config,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # GPU
        patience=10,
        save=True,
        verbose=True
    )
    
    print("\n✅ Training completed!")
    print(f"   Results: runs/detect/train/")
    
    return True

if __name__ == "__main__":
    train_basil_model()

