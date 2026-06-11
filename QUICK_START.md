# 🚀 QUICK START - BASIL YOLOv8 TRAINING

## 📍 Current Location
```
/Users/iustiniuliandodita/sera-hidroponica/
```

---

## 📊 IMAGINI DISPONIBILE ACUM

### ✅ Sample Dataset (50 imagini - Pentru Test)
```bash
basil_dataset/images/
├── train/  (30 imagini)
├── val/    (17 imagini)
└── test/   (3 imagini)
```

**Locație**: `/Users/iustiniuliandodita/sera-hidroponica/basil_dataset/images/`

**Cum să deschizi**:
```bash
# macOS
open basil_dataset/images/train/

# sau list
ls -la basil_dataset/images/train/
```

---

## 🔗 REAL DATASET (1,670 imagini - Production)

**Unde să descarci**:
```
https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs
```

**Steps**:
1. Click link de mai sus
2. Click "Download" (green button)
3. Select "YOLOv8" format
4. Download ZIP

**După download**:
```bash
# Extract
unzip basil-health-detection-1.zip -d ~/Downloads/roboflow/

# Copy to project
cp -r ~/Downloads/roboflow/basil-health-detection-1/images/* basil_dataset/images/
cp -r ~/Downloads/roboflow/basil-health-detection-1/labels/* basil_dataset/labels/

# Verify
ls basil_dataset/images/train/ | wc -l  # Should show ~1167
```

---

## 🎯 TEST ANTRENAMENT (Acum - cu 50 imagini)

```bash
# Go to project
cd /Users/iustiniuliandodita/sera-hidroponica/

# Start training
python3 train_basil_complete.py
```

**Ce se va întâmpla**:
- ✅ YOLOv8 nano va fi descărcat
- ✅ Training va rula 50 epochs (1-2 minute)
- ✅ Rezultate în: `runs/detect/train/`

---

## 📈 PRODUCTION TRAINING (După import Roboflow data)

```bash
# Cu 1,670 imagini reale
python3 train_basil_complete.py
```

**Timing**: 
- ~10-30 minute (pe GPU)
- ~1-2 ore (pe CPU)

---

## 📁 STRUCTURE FINAL

```
sera-hidroponica/
├── basil_dataset/
│   ├── images/
│   │   ├── train/  (30 now, 1,167 after import)
│   │   ├── val/    (17 now, 334 after import)
│   │   └── test/   (3 now, 169 after import)
│   ├── labels/     (same split)
│   └── data.yaml
├── train_basil_complete.py  ⭐ Run this
├── verify_dataset.py
└── create_sample_dataset.py
```

---

## ✨ WHAT'S NEXT

### Opțiunea 1: TEST ACUM
```bash
python3 train_basil_complete.py
# Quick 5-epoch test with sample data
```

### Opțiunea 2: IMPORT REAL DATA FIRST
```bash
# Download from Roboflow (see link above)
# Extract and copy as shown above
python3 train_basil_complete.py
# Full training with 1,670 images
```

---

## 🎓 CLASSES (5 Disease Types)

| ID | Name | Descriere |
|----|------|-----------|
| 0 | healthy | Sănătos |
| 1 | downy_mildew | Mucegai pufos |
| 2 | bacterial_leaf_spot | Pată bacteriană |
| 3 | leaf_miner_damage | Daunele mineri |
| 4 | general_disease | Alte boli |

---

## 🔧 TROUBLESHOOTING

**Error: "data.yaml not found"**
```bash
python3 verify_dataset.py
```

**Error: "No images found"**
```bash
# Check if data was imported
ls basil_dataset/images/train/ | wc -l
```

**Want to test inference?**
```bash
python3 -c "from ultralytics import YOLO; model = YOLO('runs/detect/train/weights/best.pt'); results = model.predict(source='basil_dataset/images/train/basil_healthy_0001.jpg')"
```

---

**Status**: ✅ READY FOR TRAINING  
**Sample Data**: ✅ AVAILABLE  
**Real Data**: ⏳ DOWNLOAD FROM ROBOFLOW
