#!/bin/bash
# 🍅 Script antrenament YOLOv8 - rulează independent în Terminal

cd ~/sera-hidroponica
source venv/bin/activate

echo "=============================================="
echo "🚀 PORNIRE ANTRENAMENT YOLOV8 - 100 EPOCI"
echo "=============================================="
echo ""
echo "📋 Dataset: tomato_real (180 imagini, 6 clase)"
echo "🖥️  Device: MPS (Apple M1 Pro GPU)"
echo "⏱️  Durată estimată: 45-60 minute"
echo ""
echo "Pentru a monitoriza progresul, deschide alt terminal și rulează:"
echo "   tail -f ~/sera-hidroponica/runs/detect/tomato_v2/results.csv"
echo ""
echo "=============================================="

python -c "
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

model = YOLO('yolov8n.pt')

results = model.train(
    data='/Users/iustiniuliandodita/sera-hidroponica/tomato_real/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    device='mps',
    project='runs/detect',
    name='tomato_v2',
    exist_ok=True,
    patience=25,
    save=True,
    plots=False,
    verbose=True,
    workers=0
)

print()
print('=' * 60)
print('✅ ANTRENAMENT FINALIZAT!')
print('=' * 60)
print()
print('📁 Model salvat: runs/detect/tomato_v2/weights/best.pt')
print()
print('Pentru a genera grafice, rulează:')
print('   python generate_plots_v2.py')
"
