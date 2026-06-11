import os
import sys
from pathlib import Path

print("=" * 60)
print("🌿 SETUP BASIL DATASET - YOLOv8 Format")
print("=" * 60)

# Create directory structure
dataset_path = Path("basil_dataset")
dataset_path.mkdir(exist_ok=True)

# Create required folders
for split in ["train", "val", "test"]:
    (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)

print("✅ Directory structure created:")
print(f"   {dataset_path}/")
print(f"   ├── images/")
print(f"   │   ├── train/")
print(f"   │   ├── val/")
print(f"   │   └── test/")
print(f"   └── labels/")
print(f"       ├── train/")
print(f"       ├── val/")
print(f"       └── test/")

# Create data.yaml configuration for YOLOv8
data_yaml = f"""path: {dataset_path.absolute()}
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
"""

yaml_file = dataset_path / "data.yaml"
yaml_file.write_text(data_yaml)

print("\n✅ data.yaml created")
print(f"   Locations: {yaml_file}")

# Create README
readme = """# Basil Health Detection Dataset

This dataset contains basil leaf images for disease detection using YOLOv8.

## Classes (5 categories)
1. **healthy** - Healthy basil leaves
2. **downy_mildew** - Downy mildew disease
3. **bacterial_leaf_spot** - Bacterial leaf spot
4. **leaf_miner_damage** - Damage from leaf miners
5. **general_disease** - Other diseases

## Source
- Original: Roboflow (disease-detection-2r3sj)
- Dataset: basil-health-detection-9crhs

## Structure
```
basil_dataset/
├── images/
│   ├── train/  (training images)
│   ├── val/    (validation images)
│   └── test/   (test images)
├── labels/
│   ├── train/  (YOLO format labels)
│   ├── val/
│   └── test/
└── data.yaml   (YOLOv8 config)
```

## How to Use
1. Download images from Roboflow
2. Extract to images/ folder
3. Generate YOLO labels
4. Train: `yolo detect train data=basil_dataset/data.yaml model=yolov8n.pt epochs=100`
"""

(dataset_path / "README.md").write_text(readme)
print("✅ README.md created\n")

print("⚠️  NEXT STEPS:")
print("=" * 60)
print("1. Download dataset from Roboflow:")
print("   https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs")
print("\n2. Extract ZIP to basil_dataset/images/")
print("   The structure should be:")
print("   basil_dataset/")
print("   ├── images/")
print("   │   ├── train/")
print("   │   ├── val/")
print("   │   └── test/")
print("   └── labels/")
print("       ├── train/")
print("       ├── val/")
print("       └── test/")
print("\n3. If needed, move images from subfolder to proper locations")
print("\n4. Start training:")
print("   python train.py --dataset basil_dataset --epochs 100")
print("=" * 60)

