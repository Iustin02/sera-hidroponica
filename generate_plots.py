#!/usr/bin/env python3
"""
Vizualizare rapidă rezultate - salvează imagini PNG
"""
import os
os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive backend

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

RESULTS_CSV = "./runs/detect/runs/train/sera_test/results.csv"
MODEL_PATH = "./runs/detect/runs/train/sera_test/weights/best.pt"

print("=" * 60)
print("📊 GENERARE GRAFICE ANTRENAMENT")
print("=" * 60)

# Citește rezultatele
df = pd.read_csv(RESULTS_CSV)
df.columns = df.columns.str.strip()
epochs = df['epoch']

# Creează graficul principal
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('📈 Rezultate Antrenament YOLOv8 Nano\nSeră Hidroponică Smart - Detectare Boli Frunze', 
             fontsize=14, fontweight='bold')

# Box Loss
axes[0,0].plot(epochs, df['train/box_loss'], 'b-o', lw=2, label='Train')
axes[0,0].plot(epochs, df['val/box_loss'], 'r--s', lw=2, label='Val')
axes[0,0].set_title('📦 Box Loss', fontweight='bold')
axes[0,0].set_xlabel('Epoch')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Cls Loss
axes[0,1].plot(epochs, df['train/cls_loss'], 'b-o', lw=2, label='Train')
axes[0,1].plot(epochs, df['val/cls_loss'], 'r--s', lw=2, label='Val')
axes[0,1].set_title('🏷️ Classification Loss', fontweight='bold')
axes[0,1].set_xlabel('Epoch')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# DFL Loss
axes[0,2].plot(epochs, df['train/dfl_loss'], 'b-o', lw=2, label='Train')
axes[0,2].plot(epochs, df['val/dfl_loss'], 'r--s', lw=2, label='Val')
axes[0,2].set_title('📐 DFL Loss', fontweight='bold')
axes[0,2].set_xlabel('Epoch')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# Precision & Recall
axes[1,0].plot(epochs, df['metrics/precision(B)'], 'g-o', lw=2, label='Precision')
axes[1,0].plot(epochs, df['metrics/recall(B)'], 'm-s', lw=2, label='Recall')
axes[1,0].set_title('🎯 Precision & Recall', fontweight='bold')
axes[1,0].set_xlabel('Epoch')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_ylim([0, 1])

# mAP50
final_map50 = df['metrics/mAP50(B)'].iloc[-1]
axes[1,1].plot(epochs, df['metrics/mAP50(B)'], 'c-o', lw=2)
axes[1,1].fill_between(epochs, df['metrics/mAP50(B)'], alpha=0.3, color='cyan')
axes[1,1].set_title(f'📊 mAP@50: {final_map50:.2%}', fontweight='bold')
axes[1,1].set_xlabel('Epoch')
axes[1,1].grid(True, alpha=0.3)

# mAP50-95
final_map = df['metrics/mAP50-95(B)'].iloc[-1]
axes[1,2].plot(epochs, df['metrics/mAP50-95(B)'], color='orange', marker='o', lw=2)
axes[1,2].fill_between(epochs, df['metrics/mAP50-95(B)'], alpha=0.3, color='orange')
axes[1,2].set_title(f'📊 mAP@50-95: {final_map:.2%}', fontweight='bold')
axes[1,2].set_xlabel('Epoch')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./training_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Grafic salvat: training_results.png")
plt.close()

# Statistici
print("\n📋 STATISTICI ANTRENAMENT:")
print("-" * 40)
last = df.iloc[-1]
print(f"   Epoci:        {int(last['epoch'])}")
print(f"   Precision:    {last['metrics/precision(B)']:.2%}")
print(f"   Recall:       {last['metrics/recall(B)']:.2%}")
print(f"   mAP@50:       {last['metrics/mAP50(B)']:.2%}")
print(f"   mAP@50-95:    {last['metrics/mAP50-95(B)']:.2%}")

# Generează imagini cu detecții
print("\n" + "=" * 60)
print("🔍 GENERARE EXEMPLE DETECȚII")
print("=" * 60)

from ultralytics import YOLO
model = YOLO(MODEL_PATH)
print(f"✅ Model încărcat")

images = glob.glob("./dataset/images/val/*.jpg")[:3]

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('🔍 Exemple Detecții YOLOv8 - Boli Frunze Roșii', fontsize=14, fontweight='bold')

class_names = ['healthy', 'bacterial_spot', 'early_blight', 'late_blight']

for idx, img_path in enumerate(images):
    results = model.predict(img_path, conf=0.2, verbose=False)
    result_img = results[0].plot()[:,:,::-1]
    axes2[idx].imshow(result_img)
    axes2[idx].axis('off')
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        det_info = []
        for box in boxes:
            cls = class_names[int(box.cls[0])]
            conf = float(box.conf[0])
            det_info.append(f"{cls}: {conf:.0%}")
        axes2[idx].set_title(f"Imagine {idx+1}\n{', '.join(det_info[:2])}", fontsize=10)
    else:
        axes2[idx].set_title(f"Imagine {idx+1}\nNo detection", fontsize=10)

plt.tight_layout()
plt.savefig('./detection_examples.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Detecții salvate: detection_examples.png")
plt.close()

print("\n" + "=" * 60)
print("✅ IMAGINI GENERATE CU SUCCES!")
print("   📊 training_results.png")
print("   🔍 detection_examples.png")
print("=" * 60)
print("\n💡 Deschide imaginile cu:")
print("   open ~/sera-hidroponica/training_results.png")
print("   open ~/sera-hidroponica/detection_examples.png")
