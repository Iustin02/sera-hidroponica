#!/usr/bin/env python3
"""
Antrenament rapid YOLOv8 fără verificări de fonturi
"""
import os
os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO

print("=" * 60)
print("🌱 Antrenament YOLOv8 - 50 Epoci")
print("=" * 60)

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    device='mps',
    project='runs/train',
    name='sera_final',
    exist_ok=True,
    batch=8,
    patience=10,
    plots=True,
    workers=0  # Evităm probleme cu multiprocessing
)

print("\n✅ Antrenament finalizat!")
print(f"Model salvat în: {results.save_dir}")
