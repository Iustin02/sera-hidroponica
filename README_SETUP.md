# 🌿 BASIL YOLOv8 SETUP - COMPLETE GUIDE

## ✅ WHAT HAS BEEN SETUP

### 📁 Project Structure Created
```
/Users/iustiniuliandodita/sera-hidroponica/

✅ basil_dataset/
   ├── images/
   │   ├── train/     (30 sample images + 1,167 after Roboflow import)
   │   ├── val/       (17 sample images + 334 after import)
   │   └── test/      (3 sample images + 169 after import)
   ├── labels/        (YOLO annotations)
   └── data.yaml      (YOLOv8 configuration)

✅ Scripts Created
   ├── train_quick.py              (5-epoch quick test)
   ├── train_basil_complete.py     (50-epoch full training)
   ├── verify_dataset.py           (Dataset verification)
   └── create_sample_dataset.py    (Generate sample images)

✅ Documentation
   ├── QUICK_START.md
   ├── DOCUMENTATION.md
   └── README_SETUP.md (this file)
```

---

## 📊 CURRENT STATUS

### ✅ Completed
- [x] Cleaned all old datasets (tomato, mint, old basil)
- [x] Created Basil-only project structure
- [x] Generated 50 sample images for testing
- [x] Created YOLOv8 configuration (data.yaml)
- [x] Created training scripts
- [x] Dataset verification script ready

### ⏳ Pending
- [ ] Download real dataset from Roboflow (1,670 images)
- [ ] Import images to basil_dataset/
- [ ] Run full training

---

## 🖼️ IMAGES LOCATION

### Sample Images (For Testing)
```bash
/Users/iustiniuliandodita/sera-hidroponica/basil_dataset/images/

├── train/   (30 images)
├── val/     (17 images)
└── test/    (3 images)
```

**Total**: 50 synthetic test images

### Real Images (After Download)
```bash
# Same location, but images replaced with:
├── train/   (1,167 real images)
├── val/     (334 real images)
└── test/    (169 real images)
```

**Total**: 1,670 real basil disease images

---

## 🔗 GET REAL IMAGES

### Link
```
https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs
```

### Steps
1. Click link above
2. Click "Download" button (green)
3. Select "YOLOv8" format
4. Download ZIP

### Import to Project

```bash
cd /Users/iustiniuliandodita/sera-hidroponica/

# Extract
unzip ~/Downloads/basil-health-detection-1.zip -d temp_roboflow/

# Copy images (replace sample images)
rm -rf basil_dataset/images/*
cp -r temp_roboflow/basil-health-detection-1/images/* basil_dataset/images/

# Copy labels
rm -rf basil_dataset/labels/*
cp -r temp_roboflow/basil-health-detection-1/labels/* basil_dataset/labels/

# Verify
python3 verify_dataset.py
```

---

## 🎓 CLASSES (5 Disease Types)

```
0: healthy              (Sănătos)
1: downy_mildew         (Mucegai pufos)
2: bacterial_leaf_spot  (Pată bacteriană)
3: leaf_miner_damage    (Daunele mineri)
4: general_disease      (Alte boli)
```

---

## 🚀 TRAINING

### Quick Test (Sample Data - 5 epochs)
```bash
cd /Users/iustiniuliandodita/sera-hidroponica/
python3 train_quick.py
```

**Duration**: 5-10 minutes (CPU)  
**Purpose**: Test pipeline works

### Full Training (Real Data - 50 epochs)
```bash
python3 train_basil_complete.py
```

**Duration**: 1-2 hours (CPU after Roboflow data imported)  
**Purpose**: Production model

---

## 📈 RESULTS

Results saved in:
```
runs/detect/train/
├── weights/
│   ├── best.pt       ← Use this for inference
│   └── last.pt
├── results.png       ← Training plots
└── confusion_matrix.png
```

---

## 🎯 INFERENCE

After training, use model for predictions:

```python
from ultralytics import YOLO

# Load best model
model = YOLO("runs/detect/train/weights/best.pt")

# Predict on image
results = model.predict(source="basil_dataset/images/val/basil_healthy_0001.jpg")

# Get predictions
print(results[0].boxes)
```

---

## 🔧 FILES SUMMARY

| File | Purpose |
|------|---------|
| `basil_dataset/data.yaml` | YOLOv8 config |
| `train_quick.py` | Quick 5-epoch test |
| `train_basil_complete.py` | Full 50-epoch training |
| `verify_dataset.py` | Check dataset integrity |
| `create_sample_dataset.py` | Generate test images |
| `DOCUMENTATION.md` | Full documentation |
| `QUICK_START.md` | Quick reference |

---

## 💾 COMMANDS REFERENCE

```bash
# Navigate to project
cd /Users/iustiniuliandodita/sera-hidroponica/

# Verify dataset
python3 verify_dataset.py

# List images
ls basil_dataset/images/train/ | wc -l

# Quick training (test)
python3 train_quick.py

# Full training
python3 train_basil_complete.py

# Open results
open runs/detect/train/
```

---

## 🎯 NEXT STEPS (ORDER IMPORTANT)

1. **Download Roboflow Dataset**
   - Click: https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs
   - Download as YOLOv8 format

2. **Extract ZIP**
   - Extract basil-health-detection-*.zip

3. **Import to Project**
   - Copy images/ and labels/ to basil_dataset/

4. **Verify**
   - Run: `python3 verify_dataset.py`
   - Should show 1,670 images

5. **Train Model**
   - Run: `python3 train_basil_complete.py`
   - Monitor: `runs/detect/train/results.png`

6. **Test Inference**
   - Use: `runs/detect/train/weights/best.pt`

---

## ✨ PROJECT FEATURES

- ✅ Complete YOLOv8 setup
- ✅ 5 basil disease classes
- ✅ Sample data for testing (50 images)
- ✅ Ready for 1,670 real images
- ✅ CPU/GPU compatible training
- ✅ Verification tools
- ✅ Complete documentation

---

## 🆘 TROUBLESHOOTING

### Problem: "No images found"
```bash
# Check if images imported
find basil_dataset/images -type f | wc -l

# If 0: Download from Roboflow and import
```

### Problem: "data.yaml not found"
```bash
# Verify file exists
ls basil_dataset/data.yaml

# Should show:
# /Users/iustiniuliandodita/sera-hidroponica/basil_dataset/data.yaml
```

### Problem: Training too slow
```bash
# This is normal on CPU
# To speed up:
# 1. Reduce batch size: batch=8
# 2. Reduce image size: imgsz=416
# 3. Use GPU if available
```

---

## 📞 STATUS

- ✅ **Setup**: COMPLETE
- ✅ **Sample Data**: READY (50 images)
- ⏳ **Real Data**: AWAITING DOWNLOAD
- ⏳ **Training**: READY TO START

---

**Location**: `/Users/iustiniuliandodita/sera-hidroponica/`  
**Dataset**: Basil only (1,670 images from Roboflow)  
**Model**: YOLOv8 nano  
**Updated**: 2026-06-11

