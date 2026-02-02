#!/bin/bash
# Script pentru antrenament YOLOv8 în background

cd ~/sera-hidroponica
source venv/bin/activate

export MPLBACKEND=Agg

python << 'EOF'
import os
os.environ['MPLBACKEND'] = 'Agg'

from ultralytics import YOLO
from datetime import datetime

print("=" * 60)
print("🌱 ANTRENAMENT YOLOv8 - 50 EPOCI")
print(f"Start: {datetime.now()}")
print("=" * 60)

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    device='mps',
    project='runs/50epochs',
    name='sera',
    exist_ok=True,
    batch=8,
    patience=15,
    plots=False,
    workers=0,
    verbose=True
)

print()
print("=" * 60)
print("✅ ANTRENAMENT FINALIZAT!")
print(f"End: {datetime.now()}")
print(f"Model: {results.save_dir}")
print("=" * 60)

# Salvează calea
with open('model_path.txt', 'w') as f:
    f.write(str(results.save_dir))
EOF
