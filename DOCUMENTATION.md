# 📚 COMPLETE DOCUMENTATION - BASIL YOLOv8 PROJECT

## 🎯 PROJECT OVERVIEW

**Project**: Basil Disease Detection using YOLOv8  
**Location**: `/Users/iustiniuliandodita/sera-hidroponica/`  
**Focus**: Basil only (Mint abandoned)  
**Dataset**: 1,670 images from Roboflow (50 sample images available for testing)  
**Model**: YOLOv8 nano

---

## 📍 QUICK NAVIGATION

```
sera-hidroponica/
├── 📁 basil_dataset/           ← Main dataset folder
│   ├── images/
│   │   ├── train/              ← 30 sample images (test with Roboflow)
│   │   ├── val/                ← 17 sample images
│   │   └── test/               ← 3 sample images
│   └── labels/                 ← YOLO format annotations
├── 📄 data.yaml                ← YOLOv8 configuration
├── 📄 train_basil_complete.py  ← Production training script
├── 📄 train_quick.py           ← Quick test training
├── 📄 verify_dataset.py        ← Verify dataset integrity
└── 📄 QUICK_START.md           ← Quick start guide
```

---

## 🖼️ IMAGE LOCATIONS

### Current Location (Sample Data)
```
/Users/iustiniuliandodita/sera-hidroponica/basil_dataset/images/
├── train/      (30 images)
├── val/        (17 images)
└── test/       (3 images)
```

**Total**: 50 synthetic test images (for pipeline verification)

### Real Dataset Location (After Roboflow Import)
After downloading from Roboflow and extracting:
```
/Users/iustiniuliandodita/sera-hidroponica/basil_dataset/images/
├── train/      (1,167 real images)
├── val/        (334 real images)
└── test/       (169 real images)
```

**Total**: 1,670 real basil images with 5 disease classes

---

## 🌿 BASIL DISEASE CLASSES (5 Classes)

| ID | Class Name | Description |
|----|-----------|-------------|
| 0 | healthy | Healthy basil leaves |
| 1 | downy_mildew | Downy mildew (fungal disease) |
| 2 | bacterial_leaf_spot | Bacterial leaf spot disease |
| 3 | leaf_miner_damage | Damage from leaf-mining insects |
| 4 | general_disease | Other disease patterns |

---

## 📥 HOW TO GET REAL IMAGES

### Step 1: Download from Roboflow
```
https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs
```

**Instructions:**
1. Click the link above
2. Click green "Download" button
3. Select "YOLOv8" format
4. Download ZIP file

### Step 2: Extract and Import

```bash
# Extract the ZIP
unzip basil-health-detection-1.zip -d ~/Downloads/roboflow/

# Copy images
cp -r ~/Downloads/roboflow/basil-health-detection-1/images/* basil_dataset/images/

# Copy labels
cp -r ~/Downloads/roboflow/basil-health-detection-1/labels/* basil_dataset/labels/

# Verify
ls basil_dataset/images/train/ | wc -l  # Should show ~1167
```

### Step 3: Verify Import

```bash
cd /Users/iustiniuliandodita/sera-hidroponica/
python3 verify_dataset.py
```

---

## 🚀 TRAINING COMMANDS

### Quick Test (5 epochs, 50 sample images)
```bash
cd /Users/iustiniuliandodita/sera-hidroponica/
python3 train_quick.py
```

**Duration**: 1-2 minutes  
**Purpose**: Verify setup and pipeline

### Full Training (50 epochs, 1,670 real images)
```bash
python3 train_basil_complete.py
```

**Duration**: 10-30 minutes (GPU) or 1-2 hours (CPU)  
**Purpose**: Production model training

### Custom Training

```bash
python3 << 'TRAIN'
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano, small, medium, large, xlarge
results = model.train(
    data="basil_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU ID or "cpu"
    patience=20,
    save=True
)
TRAIN
```

---

## 🔍 VERIFICATION COMMANDS

### Check Dataset Structure
```bash
python3 verify_dataset.py
```

### List Images
```bash
# Training images
ls basil_dataset/images/train/ | head -10

# Count total
find basil_dataset/images -type f | wc -l
```

### Check Configuration
```bash
cat basil_dataset/data.yaml
```

---

## 📊 RESULTS LOCATION

After training, results are saved in:
```
runs/detect/train/
├── weights/
│   ├── best.pt      ← Best model (use for inference)
│   └── last.pt      ← Last model
├── results.csv      ← Training metrics
├── confusion_matrix.png
├── results.png      ← Training plots
└── events.out.tfevents
```

---

## 🎯 INFERENCE - Use Trained Model

### Basic Inference
```bash
python3 << 'INFER'
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Predict on single image
results = model.predict(source="basil_dataset/images/train/basil_healthy_0001.jpg")

# Get results
print(results[0].boxes)
INFER
```

### Batch Inference
```bash
python3 << 'BATCH'
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# Predict on folder
results = model.predict(source="basil_dataset/images/val/")

# Process results
for result in results:
    print(f"Image: {result.path}")
    print(f"Detections: {len(result.boxes)}")
BATCH
```

### Real-time Video Inference
```bash
python3 << 'VIDEO'
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# Video file
results = model.predict(source="path/to/video.mp4", conf=0.25)

# Or live camera (0 = default camera)
results = model.predict(source=0)
VIDEO
```

---

## 🛠️ TROUBLESHOOTING

### Dataset is Empty
```bash
# Check if images were imported
find basil_dataset/images -type f | wc -l

# If 0, download from Roboflow:
# https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs
```

### Training Not Starting
```bash
# Verify setup
python3 verify_dataset.py

# Check YAML
cat basil_dataset/data.yaml
```

### Out of Memory
```bash
# Reduce batch size
python3 << 'PYMEM'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.train(
    data="basil_dataset/data.yaml",
    batch=8,  # Reduced from 16
    imgsz=416  # Reduced from 640
)
PYMEM
```

### GPU Not Used
```bash
# Check GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Or use CPU explicitly
model.train(device="cpu")
```

---

## 📝 FILES CREATED

| File | Purpose |
|------|---------|
| `basil_dataset/` | Main dataset folder |
| `basil_dataset/data.yaml` | YOLOv8 configuration |
| `basil_dataset/README.md` | Dataset documentation |
| `train_basil_complete.py` | Production training script |
| `train_quick.py` | Quick test training |
| `verify_dataset.py` | Dataset verification |
| `create_sample_dataset.py` | Generate sample images |
| `roboflow_download_guide.md` | Download instructions |
| `QUICK_START.md` | Quick start guide |

---

## 📈 EXPECTED RESULTS

### With Sample Data (50 images, 5 epochs)
- **Training time**: 1-2 minutes
- **Expected mAP**: Low (for pipeline testing only)
- **Purpose**: Verify setup works

### With Real Data (1,670 images, 50 epochs)
- **Training time**: 10-30 minutes (GPU) / 1-2 hours (CPU)
- **Expected mAP**: 70-85%
- **Purpose**: Production model

---

## 🔗 USEFUL LINKS

- **Roboflow Dataset**: https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **YOLO on GitHub**: https://github.com/ultralytics/ultralytics
- **Roboflow Universe**: https://universe.roboflow.com/

---

## 📞 NEXT STEPS

1. ✅ Sample dataset created and ready
2. ⏳ Download real dataset from Roboflow
3. ⏳ Extract and import images
4. ⏳ Run full training
5. ⏳ Deploy model

---

**Status**: ✅ READY FOR TRAINING  
**Sample Data**: ✅ AVAILABLE (50 images)  
**Real Data**: ⏳ AWAITING ROBOFLOW DOWNLOAD  
**Last Updated**: 2026-06-11

