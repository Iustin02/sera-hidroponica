#!/usr/bin/env python3
"""
Antrenament YOLOv8 pentru Seră Hidroponică Smart
Rulează independent cu logging
"""
import os
import sys

# Setăm backend-ul matplotlib înainte de orice import
os.environ['MPLBACKEND'] = 'Agg'

from ultralytics import YOLO
from datetime import datetime

def main():
    print("=" * 60)
    print("🌱 SERĂ HIDROPONICĂ - Antrenament YOLOv8")
    print(f"   Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Încărcăm modelul
    print("📦 Încărcare model yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    
    # Antrenăm
    print("🚀 Pornire antrenament (50 epoci)...")
    print("-" * 60)
    
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        device='mps',
        project='runs/final',
        name='sera_complete',
        exist_ok=True,
        batch=8,
        patience=10,
        plots=False,
        workers=0,
        verbose=True
    )
    
    print("-" * 60)
    print()
    print("=" * 60)
    print("✅ ANTRENAMENT FINALIZAT!")
    print(f"   End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Model salvat: {results.save_dir}")
    print("=" * 60)
    
    return results.save_dir

if __name__ == "__main__":
    save_dir = main()
    
    # Salvăm calea în fișier
    with open('last_training_dir.txt', 'w') as f:
        f.write(str(save_dir))
