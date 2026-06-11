#!/bin/bash

echo "============================================"
echo "🌿 COMPLETE BASIL SETUP - YOLOv8 PROJECT"
echo "============================================"

# Clean old files
echo "🧹 Cleaning old datasets..."
rm -rf PlantVillage* basil_dataset_old *.zip 2>/dev/null

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p basil_dataset/{images/{train,val,test},labels/{train,val,test}}

# Create data.yaml
cat > basil_dataset/data.yaml << 'YAML'
path: /Users/iustiniuliandodita/sera-hidroponica/basil_dataset
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: healthy
  1: downy_mildew
  2: bacterial_leaf_spot
  3: leaf_miner_damage
  4: general_disease
YAML

echo "✅ data.yaml created"

# Create training configuration
cat > basil_yolo_config.yaml << 'CONFIG'
# YOLOv8 Training Configuration for Basil Disease Detection

# Data
data: basil_dataset/data.yaml

# Model
model: yolov8n.pt  # nano, small, medium, large, xlarge
imgsz: 640
epochs: 50
batch: 16
device: 0

# Training options
patience: 10
save: true
verbose: true

# Augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10
translate: 0.1
scale: 0.5
flipud: 0.5
fliplr: 0.5
CONFIG

echo "✅ basil_yolo_config.yaml created"

# Create README
cat > basil_dataset/README.md << 'README'
# Basil Disease Detection Dataset

YOLOv8 object detection dataset for basil plant disease identification.

## Classes (5 categories)
0. **healthy** - Healthy basil leaves
1. **downy_mildew** - Downy mildew fungal disease
2. **bacterial_leaf_spot** - Bacterial leaf spot disease
3. **leaf_miner_damage** - Damage from leaf-mining insects
4. **general_disease** - Other disease patterns

## Source
- Original: Roboflow (disease-detection-2r3sj/basil-health-detection)
- Link: https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs

## Structure
```
basil_dataset/
├── images/
│   ├── train/  (1,167 images)
│   ├── val/    (334 images)
│   └── test/   (169 images)
├── labels/
│   ├── train/  (YOLO format annotations)
│   ├── val/
│   └── test/
├── data.yaml   (YOLOv8 configuration)
└── README.md   (this file)
```

## Training
```bash
python train_basil.py
```

## Inference
```bash
python infer_basil.py --image path/to/image.jpg
```
README

echo "✅ basil_dataset/README.md created"

# Create training script
cat > train_basil_complete.py << 'TRAIN'
#!/usr/bin/env python3
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--break-system-packages", "-q"])
    from ultralytics import YOLO

print("=" * 60)
print("🌿 YOLOv8 BASIL DISEASE DETECTION - TRAINING")
print("=" * 60)

# Check dataset
dataset_config = Path("basil_dataset/data.yaml")
if not dataset_config.exists():
    print(f"❌ {dataset_config} not found!")
    sys.exit(1)

# Load model
print("\n📦 Loading YOLOv8 nano...")
model = YOLO("yolov8n.pt")

# Train
print("\n🚀 Starting training...")
results = model.train(
    data=str(dataset_config),
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    patience=10,
    save=True,
    verbose=True
)

print("\n✅ Training complete!")
print("📊 Results saved to: runs/detect/train/")
print("🎯 Best model: runs/detect/train/weights/best.pt")

TRAIN

chmod +x train_basil_complete.py
echo "✅ train_basil_complete.py created"

echo ""
echo "============================================"
echo "✅ SETUP COMPLETE!"
echo "============================================"
echo ""
echo "📌 Next steps:"
echo "1. Download dataset from Roboflow:"
echo "   https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs"
echo ""
echo "2. Extract to basil_dataset/images/ (keeping train/val/test structure)"
echo ""
echo "3. Extract annotations to basil_dataset/labels/"
echo ""
echo "4. Train:"
echo "   python train_basil_complete.py"
echo ""
echo "📁 Dataset location: $(pwd)/basil_dataset/"
echo "🔗 Roboflow URL: https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs"
echo "============================================"

