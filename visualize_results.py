#!/usr/bin/env python3
"""
Script pentru vizualizarea rezultatelor antrenamentului YOLOv8
și afișarea detecțiilor pe imagini de test.

Proiect: Seră Hidroponică Smart - Detectare Boli Frunze Roșii
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from PIL import Image
import glob

# Configurare matplotlib pentru afișare frumoasă
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def plot_training_results(results_csv_path, save_path=None):
    """
    Afișează graficele cu rezultatele antrenamentului.
    
    Args:
        results_csv_path: Calea către fișierul results.csv
        save_path: Calea pentru salvarea graficului (opțional)
    """
    
    print("=" * 60)
    print("📊 VIZUALIZARE REZULTATE ANTRENAMENT")
    print("=" * 60)
    
    # Încărcăm datele
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()  # Curățăm spațiile din nume coloane
    
    # Creăm figura cu subploturi
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('📈 Rezultate Antrenament YOLOv8 - Seră Hidroponică Smart', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    epochs = df['epoch']
    
    # 1. Box Loss (Train vs Val)
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['train/box_loss'], 'b-', label='Train', linewidth=2, marker='o')
    ax1.plot(epochs, df['val/box_loss'], 'r--', label='Validation', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Box Loss')
    ax1.set_title('📦 Box Loss (Localizare)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Classification Loss (Train vs Val)
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['train/cls_loss'], 'b-', label='Train', linewidth=2, marker='o')
    ax2.plot(epochs, df['val/cls_loss'], 'r--', label='Validation', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Classification Loss')
    ax2.set_title('🏷️ Classification Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. DFL Loss (Train vs Val)
    ax3 = axes[0, 2]
    ax3.plot(epochs, df['train/dfl_loss'], 'b-', label='Train', linewidth=2, marker='o')
    ax3.plot(epochs, df['val/dfl_loss'], 'r--', label='Validation', linewidth=2, marker='s')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('DFL Loss')
    ax3.set_title('📐 Distribution Focal Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision & Recall
    ax4 = axes[1, 0]
    ax4.plot(epochs, df['metrics/precision(B)'], 'g-', label='Precision', linewidth=2, marker='o')
    ax4.plot(epochs, df['metrics/recall(B)'], 'm-', label='Recall', linewidth=2, marker='s')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.set_title('🎯 Precision & Recall')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. mAP50
    ax5 = axes[1, 1]
    ax5.plot(epochs, df['metrics/mAP50(B)'], 'c-', label='mAP@50', linewidth=2, marker='o')
    ax5.fill_between(epochs, df['metrics/mAP50(B)'], alpha=0.3)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('mAP@50')
    ax5.set_title('📊 Mean Average Precision @ IoU=0.5')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # 6. mAP50-95
    ax6 = axes[1, 2]
    ax6.plot(epochs, df['metrics/mAP50-95(B)'], 'orange', label='mAP@50-95', linewidth=2, marker='o')
    ax6.fill_between(epochs, df['metrics/mAP50-95(B)'], alpha=0.3, color='orange')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('mAP@50-95')
    ax6.set_title('📊 Mean Average Precision @ IoU=0.5:0.95')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grafic salvat: {save_path}")
    
    plt.show()
    
    # Afișăm statistici finale
    print("\n" + "=" * 60)
    print("📋 STATISTICI FINALE (ultima epocă):")
    print("=" * 60)
    last = df.iloc[-1]
    print(f"   ├── Box Loss (train):    {last['train/box_loss']:.4f}")
    print(f"   ├── Cls Loss (train):    {last['train/cls_loss']:.4f}")
    print(f"   ├── Precision:           {last['metrics/precision(B)']:.4f}")
    print(f"   ├── Recall:              {last['metrics/recall(B)']:.4f}")
    print(f"   ├── mAP@50:              {last['metrics/mAP50(B)']:.4f}")
    print(f"   └── mAP@50-95:           {last['metrics/mAP50-95(B)']:.4f}")


def show_detection_samples(model_path, images_dir, num_samples=3, save_dir=None):
    """
    Afișează detecții pe imagini de test.
    
    Args:
        model_path: Calea către modelul antrenat (.pt)
        images_dir: Directorul cu imagini de test
        num_samples: Numărul de imagini de afișat
        save_dir: Director pentru salvarea imaginilor cu detecții
    """
    
    print("\n" + "=" * 60)
    print("🔍 DETECȚII PE IMAGINI DE TEST")
    print("=" * 60)
    
    # Încărcăm modelul
    model = YOLO(model_path)
    print(f"✅ Model încărcat: {model_path}")
    
    # Găsim imagini
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(images_dir, ext)))
    
    if not images:
        print(f"⚠️  Nu am găsit imagini în {images_dir}")
        return
    
    # Selectăm random câteva imagini
    selected = images[:min(num_samples, len(images))]
    
    # Creăm figura
    fig, axes = plt.subplots(1, len(selected), figsize=(6*len(selected), 6))
    if len(selected) == 1:
        axes = [axes]
    
    fig.suptitle('🔍 Detecții YOLOv8 - Boli Frunze Roșii', fontsize=14, fontweight='bold')
    
    for idx, img_path in enumerate(selected):
        # Rulăm predicția
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        # Obținem imaginea cu bounding boxes desenate
        result_img = results[0].plot()
        
        # Convertim BGR la RGB
        result_img = result_img[:, :, ::-1]
        
        axes[idx].imshow(result_img)
        axes[idx].set_title(f'Imagine {idx+1}: {os.path.basename(img_path)}')
        axes[idx].axis('off')
        
        # Afișăm detecțiile găsite
        boxes = results[0].boxes
        if len(boxes) > 0:
            print(f"\n📸 {os.path.basename(img_path)}:")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                print(f"   └── {cls_name}: {conf:.2%}")
        else:
            print(f"\n📸 {os.path.basename(img_path)}: Nicio detecție")
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'detection_samples.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Imagini salvate: {save_path}")
    
    plt.show()


def main():
    """
    Funcția principală - afișează graficele și detecțiile.
    """
    
    # Căi către rezultate
    results_dir = "./runs/detect/runs/train/sera_test"
    results_csv = os.path.join(results_dir, "results.csv")
    model_path = os.path.join(results_dir, "weights", "best.pt")
    images_dir = "./dataset/images/val"
    
    # Verificăm dacă există rezultate
    if not os.path.exists(results_csv):
        print("❌ Nu am găsit results.csv!")
        print("   Rulează mai întâi antrenamentul.")
        return
    
    # 1. Afișăm graficele
    plot_training_results(results_csv, save_path=os.path.join(results_dir, "training_plots.png"))
    
    # 2. Afișăm detecții
    if os.path.exists(model_path) and os.path.exists(images_dir):
        show_detection_samples(model_path, images_dir, num_samples=3, save_dir=results_dir)
    else:
        print("\n⚠️  Model sau imagini de test lipsă pentru afișarea detecțiilor.")


if __name__ == "__main__":
    main()
