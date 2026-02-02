#!/usr/bin/env python3
"""
Script pentru antrenament complet YOLOv8 + vizualizare rezultate
Proiect: Seră Hidroponică Smart
"""

from ultralytics import YOLO
import os
import matplotlib
matplotlib.use('TkAgg')  # Backend pentru afișare grafice
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob

# =============================================================================
# CONFIGURARE
# =============================================================================
EPOCHS = 50
BATCH_SIZE = 8
IMAGE_SIZE = 640
DEVICE = 'mps'
PROJECT_NAME = 'runs/train'
EXPERIMENT_NAME = 'sera_50epochs'

# =============================================================================
# ANTRENAMENT
# =============================================================================
def train_model():
    """Antrenează modelul YOLOv8."""
    
    print("=" * 70)
    print("🌱 SERĂ HIDROPONICĂ - Antrenament YOLOv8 (50 Epoci)")
    print("=" * 70)
    print()
    
    # Încarcă modelul
    model = YOLO('yolov8n.pt')
    
    # Antrenează
    results = model.train(
        data='data.yaml',
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        verbose=True,
        batch=BATCH_SIZE,
        patience=10,
        plots=True,
        save=True
    )
    
    print()
    print("=" * 70)
    print("✅ Antrenament finalizat!")
    print("=" * 70)
    
    return results


# =============================================================================
# VIZUALIZARE GRAFICE
# =============================================================================
def plot_results():
    """Afișează graficele cu rezultatele antrenamentului."""
    
    results_dir = f"./runs/detect/{PROJECT_NAME}/{EXPERIMENT_NAME}"
    results_csv = os.path.join(results_dir, "results.csv")
    
    if not os.path.exists(results_csv):
        # Încearcă calea alternativă
        results_dir = f"./runs/detect/runs/train/{EXPERIMENT_NAME}"
        results_csv = os.path.join(results_dir, "results.csv")
    
    if not os.path.exists(results_csv):
        print(f"❌ Nu am găsit results.csv în {results_csv}")
        return None
    
    print("\n📊 Generare grafice...")
    
    # Încarcă datele
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # Creăm figura
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('📈 Rezultate Antrenament YOLOv8 - Seră Hidroponică Smart\n50 Epoci pe GPU M1 Pro', 
                 fontsize=14, fontweight='bold')
    
    epochs = df['epoch']
    
    # 1. Box Loss
    axes[0, 0].plot(epochs, df['train/box_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, df['val/box_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('📦 Box Loss (Localizare)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Classification Loss
    axes[0, 1].plot(epochs, df['train/cls_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, df['val/cls_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('🏷️ Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. DFL Loss
    axes[0, 2].plot(epochs, df['train/dfl_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, df['val/dfl_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('📐 DFL Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Precision & Recall
    axes[1, 0].plot(epochs, df['metrics/precision(B)'], 'g-', label='Precision', linewidth=2)
    axes[1, 0].plot(epochs, df['metrics/recall(B)'], 'm-', label='Recall', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('🎯 Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    # 5. mAP50
    axes[1, 1].plot(epochs, df['metrics/mAP50(B)'], 'c-', linewidth=2)
    axes[1, 1].fill_between(epochs, df['metrics/mAP50(B)'], alpha=0.3, color='cyan')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mAP@50')
    axes[1, 1].set_title(f'📊 mAP@50 (Final: {df["metrics/mAP50(B)"].iloc[-1]:.2%})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    # 6. mAP50-95
    axes[1, 2].plot(epochs, df['metrics/mAP50-95(B)'], 'orange', linewidth=2)
    axes[1, 2].fill_between(epochs, df['metrics/mAP50-95(B)'], alpha=0.3, color='orange')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('mAP@50-95')
    axes[1, 2].set_title(f'📊 mAP@50-95 (Final: {df["metrics/mAP50-95(B)"].iloc[-1]:.2%})')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Salvează graficul
    save_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Grafic salvat: {save_path}")
    
    # Afișează statistici
    print("\n" + "=" * 70)
    print("📋 STATISTICI FINALE:")
    print("=" * 70)
    last = df.iloc[-1]
    print(f"   ├── Precision:    {last['metrics/precision(B)']:.2%}")
    print(f"   ├── Recall:       {last['metrics/recall(B)']:.2%}")
    print(f"   ├── mAP@50:       {last['metrics/mAP50(B)']:.2%}")
    print(f"   └── mAP@50-95:    {last['metrics/mAP50-95(B)']:.2%}")
    
    plt.show()
    
    return results_dir


# =============================================================================
# DETECȚII PE IMAGINI
# =============================================================================
def show_detections(results_dir):
    """Afișează detecții pe imagini de test."""
    
    model_path = os.path.join(results_dir, "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"❌ Model nu a fost găsit: {model_path}")
        return
    
    print("\n🔍 Generare detecții pe imagini de test...")
    
    # Încarcă modelul
    model = YOLO(model_path)
    
    # Găsește imagini de test
    images = glob.glob("./dataset/images/val/*.jpg")[:3]
    
    if not images:
        print("⚠️ Nu am găsit imagini de test")
        return
    
    # Creăm figura
    fig, axes = plt.subplots(1, len(images), figsize=(6*len(images), 6))
    if len(images) == 1:
        axes = [axes]
    
    fig.suptitle('🔍 Detecții YOLOv8 - Boli Frunze Roșii', fontsize=14, fontweight='bold')
    
    for idx, img_path in enumerate(images):
        results = model.predict(img_path, conf=0.25, verbose=False)
        result_img = results[0].plot()
        result_img = result_img[:, :, ::-1]  # BGR to RGB
        
        axes[idx].imshow(result_img)
        axes[idx].set_title(f'Imagine {idx+1}')
        axes[idx].axis('off')
        
        # Afișează detecțiile
        boxes = results[0].boxes
        if len(boxes) > 0:
            print(f"\n📸 Imagine {idx+1}:")
            for box in boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"   └── {cls_name}: {conf:.2%}")
    
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, "detections.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Detecții salvate: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--viz-only':
        # Doar vizualizare (fără antrenament)
        results_dir = plot_results()
        if results_dir:
            show_detections(results_dir)
    else:
        # Antrenament complet + vizualizare
        train_model()
        results_dir = plot_results()
        if results_dir:
            show_detections(results_dir)
