#!/usr/bin/env python3
"""
🍅 ANTRENAMENT YOLOV8 - 100 EPOCI
Rulează în background fără întreruperi
"""

import os
import sys

# IMPORTANT: Setează backend non-interactiv ÎNAINTE de orice import
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("🚀 ANTRENAMENT YOLOV8 NANO - 100 EPOCI")
    print("=" * 60)
    
    data_yaml = "/Users/iustiniuliandodita/sera-hidroponica/tomato_real/data.yaml"
    
    # Verifică dacă dataset-ul există
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset nu există: {data_yaml}")
        sys.exit(1)
    
    print(f"\n📋 Configurație:")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epoci: 100")
    print(f"   Device: MPS")
    print(f"   Batch: 8")
    
    # Încarcă modelul
    model = YOLO('yolov8n.pt')
    
    print("\n⏳ Pornesc antrenamentul...\n")
    
    # Antrenare
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=8,  # Batch mai mic pentru stabilitate
        device='mps',
        project='runs/detect',
        name='tomato_v2',
        exist_ok=True,
        patience=25,
        save=True,
        plots=False,
        verbose=True,
        workers=0  # Evită probleme multiprocessing
    )
    
    print("\n" + "=" * 60)
    print("✅ ANTRENAMENT FINALIZAT!")
    print("=" * 60)
    
    # Afișează rezultatele
    results_path = "runs/detect/tomato_v2/results.csv"
    if os.path.exists(results_path):
        import pandas as pd
        df = pd.read_csv(results_path)
        last = df.iloc[-1]
        print(f"\n📊 Rezultate finale:")
        print(f"   Precision: {last['metrics/precision(B)']:.2%}")
        print(f"   Recall: {last['metrics/recall(B)']:.2%}")
        print(f"   mAP@50: {last['metrics/mAP50(B)']:.2%}")
        print(f"   mAP@50-95: {last['metrics/mAP50-95(B)']:.2%}")
    
    print(f"\n📁 Model salvat în: runs/detect/tomato_v2/weights/best.pt")

if __name__ == "__main__":
    main()
