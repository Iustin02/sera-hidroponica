#!/usr/bin/env python3
"""
Vizualizare rezultate antrenament YOLOv8
Proiect: Seră Hidroponică Smart
"""
import os
os.environ['MPLBACKEND'] = 'TkAgg'  # Pentru afișare interactivă

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

# Calea către rezultatele existente
RESULTS_DIR = "./runs/detect/runs/train/sera_test"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

def plot_training_curves():
    """Generează grafice cu curbe de antrenament."""
    
    print("=" * 60)
    print("📊 VIZUALIZARE REZULTATE ANTRENAMENT")
    print("=" * 60)
    
    # Încarcă datele
    df = pd.read_csv(RESULTS_CSV)
    df.columns = df.columns.str.strip()
    
    epochs = df['epoch']
    
    # Creează figura cu 6 subploturi
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('📈 Rezultate Antrenament YOLOv8 Nano\nSeră Hidroponică Smart - Detectare Boli Frunze Roșii', 
                 fontsize=14, fontweight='bold')
    
    # Culorile
    train_color = '#2196F3'  # Albastru
    val_color = '#F44336'    # Roșu
    
    # 1. Box Loss
    ax = axes[0, 0]
    ax.plot(epochs, df['train/box_loss'], color=train_color, linewidth=2, marker='o', label='Train')
    ax.plot(epochs, df['val/box_loss'], color=val_color, linewidth=2, marker='s', linestyle='--', label='Validation')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('📦 Box Loss (Localizare)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    # 2. Classification Loss
    ax = axes[0, 1]
    ax.plot(epochs, df['train/cls_loss'], color=train_color, linewidth=2, marker='o', label='Train')
    ax.plot(epochs, df['val/cls_loss'], color=val_color, linewidth=2, marker='s', linestyle='--', label='Validation')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('🏷️ Classification Loss', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    # 3. DFL Loss
    ax = axes[0, 2]
    ax.plot(epochs, df['train/dfl_loss'], color=train_color, linewidth=2, marker='o', label='Train')
    ax.plot(epochs, df['val/dfl_loss'], color=val_color, linewidth=2, marker='s', linestyle='--', label='Validation')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('📐 Distribution Focal Loss', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    # 4. Precision & Recall
    ax = axes[1, 0]
    ax.plot(epochs, df['metrics/precision(B)'], color='#4CAF50', linewidth=2, marker='o', label='Precision')
    ax.plot(epochs, df['metrics/recall(B)'], color='#9C27B0', linewidth=2, marker='s', label='Recall')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('🎯 Precision & Recall', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.set_facecolor('#f8f9fa')
    
    # 5. mAP@50
    ax = axes[1, 1]
    ax.plot(epochs, df['metrics/mAP50(B)'], color='#00BCD4', linewidth=2, marker='o')
    ax.fill_between(epochs, df['metrics/mAP50(B)'], alpha=0.3, color='#00BCD4')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('mAP@50', fontsize=10)
    final_map50 = df['metrics/mAP50(B)'].iloc[-1]
    ax.set_title(f'📊 mAP@IoU=0.5 (Final: {final_map50:.2%})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(0.1, df['metrics/mAP50(B)'].max() * 1.2)])
    ax.set_facecolor('#f8f9fa')
    
    # 6. mAP@50-95
    ax = axes[1, 2]
    ax.plot(epochs, df['metrics/mAP50-95(B)'], color='#FF9800', linewidth=2, marker='o')
    ax.fill_between(epochs, df['metrics/mAP50-95(B)'], alpha=0.3, color='#FF9800')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('mAP@50-95', fontsize=10)
    final_map = df['metrics/mAP50-95(B)'].iloc[-1]
    ax.set_title(f'📊 mAP@IoU=0.5:0.95 (Final: {final_map:.2%})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(0.1, df['metrics/mAP50-95(B)'].max() * 1.2)])
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Salvează graficul
    save_path = os.path.join(RESULTS_DIR, "training_curves_pyplot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Grafic salvat: {save_path}")
    
    # Afișează statistici
    print("\n" + "-" * 60)
    print("📋 STATISTICI ANTRENAMENT (Ultima Epocă):")
    print("-" * 60)
    last = df.iloc[-1]
    print(f"   ├── Epoci antrenate:     {int(last['epoch'])}")
    print(f"   ├── Box Loss (train):    {last['train/box_loss']:.4f}")
    print(f"   ├── Cls Loss (train):    {last['train/cls_loss']:.4f}")
    print(f"   ├── Precision:           {last['metrics/precision(B)']:.2%}")
    print(f"   ├── Recall:              {last['metrics/recall(B)']:.2%}")
    print(f"   ├── mAP@50:              {last['metrics/mAP50(B)']:.2%}")
    print(f"   └── mAP@50-95:           {last['metrics/mAP50-95(B)']:.2%}")
    
    plt.show()
    return df


def show_detections():
    """Afișează imagini cu detecții."""
    
    print("\n" + "=" * 60)
    print("🔍 EXEMPLE DETECȚII PE IMAGINI")
    print("=" * 60)
    
    # Verifică dacă modelul există
    model_path = os.path.join(RESULTS_DIR, "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"⚠️  Model nu există: {model_path}")
        return
    
    # Încarcă modelul
    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"✅ Model încărcat: {model_path}")
    
    # Găsește imagini
    images = glob.glob("./dataset/images/val/*.jpg")[:3]
    if not images:
        print("⚠️  Nu am găsit imagini în dataset/images/val/")
        return
    
    # Creează figura
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    if n_images == 1:
        axes = [axes]
    
    fig.suptitle('🔍 Detecții YOLOv8 - Boli Frunze Roșii\nSeră Hidroponică Smart', 
                 fontsize=14, fontweight='bold')
    
    class_names = {
        0: 'healthy (sănătos)',
        1: 'bacterial_spot (pată bacteriană)',
        2: 'early_blight (mană timpurie)',
        3: 'late_blight (mană târzie)'
    }
    
    for idx, img_path in enumerate(images):
        # Rulează predicția
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        # Obține imaginea cu bounding boxes
        result_img = results[0].plot()
        result_img = result_img[:, :, ::-1]  # BGR -> RGB
        
        axes[idx].imshow(result_img)
        axes[idx].axis('off')
        
        # Afișează detecțiile
        boxes = results[0].boxes
        if len(boxes) > 0:
            detections = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = class_names.get(cls_id, f'class_{cls_id}')
                detections.append(f"{cls_name}: {conf:.0%}")
            title = f"Imagine {idx+1}\n" + "\n".join(detections[:2])
        else:
            title = f"Imagine {idx+1}\nNicio detecție"
        
        axes[idx].set_title(title, fontsize=9)
    
    plt.tight_layout()
    
    # Salvează
    save_path = os.path.join(RESULTS_DIR, "detection_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Detecții salvate: {save_path}")
    
    plt.show()


def main():
    """Funcția principală."""
    
    if not os.path.exists(RESULTS_CSV):
        print(f"❌ Fișierul results.csv nu există: {RESULTS_CSV}")
        print("   Rulează mai întâi antrenamentul.")
        return
    
    # 1. Grafice antrenament
    plot_training_curves()
    
    # 2. Exemple detecții
    show_detections()
    
    print("\n" + "=" * 60)
    print("✅ VIZUALIZARE COMPLETĂ!")
    print("=" * 60)


if __name__ == "__main__":
    main()
