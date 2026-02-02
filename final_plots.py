#!/usr/bin/env python3
"""
📊 GENERARE GRAFICE FINALE - ANTRENAMENT 100 EPOCI
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Calea către rezultate
RESULTS_PATH = Path.home() / "sera-hidroponica/runs/detect/runs/detect/tomato_v2/results.csv"
MODEL_PATH = Path.home() / "sera-hidroponica/runs/detect/runs/detect/tomato_v2/weights/best.pt"
VAL_IMAGES = Path.home() / "sera-hidroponica/tomato_real/images/val"
OUTPUT_DIR = Path.home() / "sera-hidroponica"

print("=" * 60)
print("📊 GENERARE GRAFICE ANTRENAMENT - 100 EPOCI")
print("=" * 60)

# Citește rezultatele
df = pd.read_csv(RESULTS_PATH)
df.columns = df.columns.str.strip()

# Creează figura cu 6 subploturi
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('YOLOv8 Nano - Antrenament 100 Epoci\nDetecție Boli Frunze Roșii', fontsize=14, fontweight='bold')

epochs = df['epoch']

# 1. Box Loss
ax1 = axes[0, 0]
ax1.plot(epochs, df['train/box_loss'], 'b-', label='Train', linewidth=2)
ax1.plot(epochs, df['val/box_loss'], 'r--', label='Val', linewidth=2)
ax1.set_title('Box Loss', fontweight='bold')
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Class Loss
ax2 = axes[0, 1]
ax2.plot(epochs, df['train/cls_loss'], 'b-', label='Train', linewidth=2)
ax2.plot(epochs, df['val/cls_loss'], 'r--', label='Val', linewidth=2)
ax2.set_title('Classification Loss', fontweight='bold')
ax2.set_xlabel('Epoca')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. DFL Loss
ax3 = axes[0, 2]
ax3.plot(epochs, df['train/dfl_loss'], 'b-', label='Train', linewidth=2)
ax3.plot(epochs, df['val/dfl_loss'], 'r--', label='Val', linewidth=2)
ax3.set_title('DFL Loss', fontweight='bold')
ax3.set_xlabel('Epoca')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Precision & Recall
ax4 = axes[1, 0]
ax4.plot(epochs, df['metrics/precision(B)'] * 100, 'g-', label='Precision', linewidth=2)
ax4.plot(epochs, df['metrics/recall(B)'] * 100, 'm-', label='Recall', linewidth=2)
ax4.set_title('Precision & Recall', fontweight='bold')
ax4.set_xlabel('Epoca')
ax4.set_ylabel('%')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 100])

# 5. mAP@50
ax5 = axes[1, 1]
ax5.plot(epochs, df['metrics/mAP50(B)'] * 100, 'c-', linewidth=2)
ax5.fill_between(epochs, 0, df['metrics/mAP50(B)'] * 100, alpha=0.3)
ax5.set_title('mAP@50', fontweight='bold')
ax5.set_xlabel('Epoca')
ax5.set_ylabel('%')
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0, 100])

# Adaugă valoarea finală
final_map50 = df['metrics/mAP50(B)'].iloc[-1] * 100
ax5.axhline(y=final_map50, color='r', linestyle='--', alpha=0.7)
ax5.text(epochs.iloc[-1] * 0.7, final_map50 + 3, f'Final: {final_map50:.1f}%', fontsize=10, color='r')

# 6. mAP@50-95
ax6 = axes[1, 2]
ax6.plot(epochs, df['metrics/mAP50-95(B)'] * 100, 'orange', linewidth=2)
ax6.fill_between(epochs, 0, df['metrics/mAP50-95(B)'] * 100, alpha=0.3, color='orange')
ax6.set_title('mAP@50-95', fontweight='bold')
ax6.set_xlabel('Epoca')
ax6.set_ylabel('%')
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 100])

# Adaugă valoarea finală
final_map = df['metrics/mAP50-95(B)'].iloc[-1] * 100
ax6.axhline(y=final_map, color='r', linestyle='--', alpha=0.7)
ax6.text(epochs.iloc[-1] * 0.7, final_map + 3, f'Final: {final_map:.1f}%', fontsize=10, color='r')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_results_100epochs.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ Grafic salvat: training_results_100epochs.png")

# Generează exemple de detecție
print("\n📸 Generare exemple detecție...")

from ultralytics import YOLO

model = YOLO(str(MODEL_PATH))

# Găsește imaginile de validare
val_images = list(VAL_IMAGES.glob("*.jpg"))[:6]

if val_images:
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    fig2.suptitle('Exemple Detecție - Model Antrenat 100 Epoci', fontsize=14, fontweight='bold')
    
    for idx, img_path in enumerate(val_images):
        ax = axes2[idx // 3, idx % 3]
        
        # Rulează inferența
        results = model(str(img_path), verbose=False)
        
        # Obține imaginea cu detecții
        result_img = results[0].plot()
        result_img = result_img[:, :, ::-1]  # BGR to RGB
        
        ax.imshow(result_img)
        ax.set_title(f'Imagine {idx + 1}', fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detection_examples_100epochs.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Exemple detecție salvate: detection_examples_100epochs.png")

# Afișează statistici finale
print("\n" + "=" * 60)
print("📊 STATISTICI FINALE ANTRENAMENT")
print("=" * 60)

best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
best_map50 = df['metrics/mAP50(B)'].max() * 100
best_map = df['metrics/mAP50-95(B)'].max() * 100

print(f"""
   Epoci totale:     100
   Cea mai bună epocă: {best_epoch}
   
   📈 Metrici finale (epoca 100):
   ─────────────────────────────
   Precision:        {df['metrics/precision(B)'].iloc[-1]*100:.1f}%
   Recall:           {df['metrics/recall(B)'].iloc[-1]*100:.1f}%
   mAP@50:           {df['metrics/mAP50(B)'].iloc[-1]*100:.1f}%
   mAP@50-95:        {df['metrics/mAP50-95(B)'].iloc[-1]*100:.1f}%
   
   🏆 Cele mai bune rezultate:
   ─────────────────────────────
   mAP@50 max:       {best_map50:.1f}%
   mAP@50-95 max:    {best_map:.1f}%
   
   📁 Model salvat:
   {MODEL_PATH}
""")

print("=" * 60)
print("✅ GRAFICE GENERATE CU SUCCES!")
print("=" * 60)
print(f"\n   Deschide cu:")
print(f"   open ~/sera-hidroponica/training_results_100epochs.png")
print(f"   open ~/sera-hidroponica/detection_examples_100epochs.png")
