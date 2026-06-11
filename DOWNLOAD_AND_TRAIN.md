# 🌿 BASIL YOLOv8 - DOWNLOAD & TRAINING GUIDE

## 📋 QUICK STEPS (3 COMMANDS)

```bash
# Step 1: Download & Import
cd /Users/iustiniuliandodita/sera-hidroponica/
python3 auto_import_roboflow.py

# Step 2: Full Training (1-2 hours)
python3 train_basil_complete.py

# Step 3: View Results
open runs/detect/train/
```

---

## 📥 DETAILED DOWNLOAD INSTRUCTIONS

### Step 1: Download from Roboflow

1. **Open this link in browser:**
   ```
   https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs
   ```

2. **Click "Download" button** (green, top-right corner)

3. **Select "YOLOv8" format** from dropdown menu

4. **Click "Download"** button to save ZIP file

5. **Wait for download** to complete (1,670 images ≈ 300-500 MB)

6. **ZIP file location:** 
   ```
   ~/Downloads/basil-health-detection-*.zip
   ```

### Step 2: Run Auto-Import

After download completes, run:

```bash
cd /Users/iustiniuliandodita/sera-hidroponica/
python3 auto_import_roboflow.py
```

The script will:
- ✅ Find the ZIP file in ~/Downloads/
- ✅ Extract it
- ✅ Copy images to basil_dataset/images/
- ✅ Copy labels to basil_dataset/labels/
- ✅ Verify the dataset
- ✅ Clean up temporary files
- ✅ Show you the summary

**Expected output:**
```
✅ Dataset import successful!

📊 Summary:
   Train images: 1,169
   Val images:   334
   Test images:  169
   Total:        1,672
```

---

## 🚀 FULL TRAINING

After import is complete, run:

```bash
python3 train_basil_complete.py
```

### Training Details

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8 nano |
| Epochs | 50 |
| Batch Size | 16 |
| Image Size | 640x640 |
| Device | CPU (auto-detects GPU) |
| Duration | 1-2 hours (CPU) |
| Output | runs/detect/train/ |

### Monitor Training

While training runs, you can monitor:

```bash
# In another terminal:
open runs/detect/train/
```

The folder contains:
- `results.png` - Training/validation plots
- `confusion_matrix.png` - Accuracy per disease class
- `weights/best.pt` - Best model (use this for inference)
- `weights/last.pt` - Last epoch model
- `events.out.tfevents.*` - TensorBoard logs

### Training Stops When

- ✅ 50 epochs complete
- ⚠️ Validation loss stops improving (patience=10 epochs)
- ❌ Manual stop (Ctrl+C)

---

## 🎯 AFTER TRAINING

### Use the Model for Predictions

```python
from ultralytics import YOLO

# Load best model
model = YOLO("runs/detect/train/weights/best.pt")

# Predict on image
results = model.predict(source="basil_dataset/images/val/", conf=0.25)

# Get predictions
for result in results:
    print(result.boxes)
```

### Check Model Performance

Results saved in `runs/detect/train/`:

```bash
# View results
open runs/detect/train/results.png

# View confusion matrix
open runs/detect/train/confusion_matrix.png
```

---

## 🆘 TROUBLESHOOTING

### Problem: "ZIP file not found"

**Solution:**
1. Make sure you downloaded from the correct Roboflow link
2. Verify ZIP is in ~/Downloads/
3. Check filename: should start with "basil-health-detection-"

```bash
# List downloads
ls ~/Downloads/basil-health-detection-*.zip
```

### Problem: "No images found after import"

**Solution:**
1. Check if ZIP was properly downloaded
2. Verify extraction: `ls basil_dataset/images/`
3. Try downloading again

### Problem: "Training is very slow"

**Solution:**
- This is normal on CPU (1-2 hours is expected)
- Each epoch takes 2-4 minutes on CPU
- To speed up: reduce batch size or image size
  
Edit `train_basil_complete.py`:
```python
results = model.train(
    batch=8,        # Reduce from 16
    imgsz=416,      # Reduce from 640
    # ... rest of config
)
```

### Problem: "CUDA/GPU errors"

**Solution:**
- Scripts already configured for CPU
- GPU will auto-detect if available
- If errors appear, just let training continue on CPU

### Problem: "Out of memory"

**Solution:**
1. Reduce batch size: `batch=8` or `batch=4`
2. Reduce image size: `imgsz=416`
3. Close other applications

---

## 📊 DISEASE CLASSES

The model will detect 5 basil diseases:

```
Class 0: healthy              (Sănătos)
Class 1: downy_mildew         (Mucegai pufos) 
Class 2: bacterial_leaf_spot  (Pată bacteriană)
Class 3: leaf_miner_damage    (Daunele mineri)
Class 4: general_disease      (Alte boli)
```

---

## ⏱️ TIME ESTIMATES

| Step | Time |
|------|------|
| Download ZIP | 5-15 min (depends on internet) |
| Extract & Import | 2-5 min |
| Verification | 1 min |
| **Full Training** | **1-2 hours** |
| **Total** | **1.5-2.5 hours** |

---

## 💾 STORAGE REQUIREMENTS

| Item | Size |
|------|------|
| Downloaded ZIP | 300-500 MB |
| Extracted dataset | 1.5-2 GB |
| Trained model | 100 MB |
| **Total needed** | **2-3 GB** |

---

## ✅ COMPLETE WORKFLOW

```
1. Download ZIP from Roboflow (15 min)
   └─ https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs

2. Run auto-import (5 min)
   └─ python3 auto_import_roboflow.py

3. Verify dataset imported (1 min)
   └─ Script runs verify_dataset.py automatically

4. Train model (1-2 hours)
   └─ python3 train_basil_complete.py

5. View results (1 min)
   └─ open runs/detect/train/

6. Use model for predictions (ongoing)
   └─ Load best.pt and run inference
```

---

## 🔗 KEY COMMANDS

```bash
# Navigate to project
cd /Users/iustiniuliandodita/sera-hidroponica/

# Import downloaded dataset
python3 auto_import_roboflow.py

# Train model (production)
python3 train_basil_complete.py

# Quick test (5 epochs - optional)
python3 train_quick.py

# Verify dataset
python3 verify_dataset.py

# View all training images
open basil_dataset/images/

# View training results
open runs/detect/train/

# Check downloaded file
ls -lh ~/Downloads/basil-health-detection-*.zip
```

---

## 📞 STATUS CHECK

Before running, verify:

```bash
# Check project structure
ls basil_dataset/

# Verify config
cat basil_dataset/data.yaml

# Check sample data
ls basil_dataset/images/train/ | wc -l
```

Expected:
- ✅ basil_dataset/ directory exists
- ✅ images/ subdirectory exists  
- ✅ labels/ subdirectory exists
- ✅ data.yaml file exists
- ✅ Currently 30 train images (sample data)

---

## 🎉 YOU'RE READY!

1. **Download the ZIP** from Roboflow link above
2. **Run:** `python3 auto_import_roboflow.py`
3. **Wait:** Auto-import processes your data (5 min)
4. **Train:** `python3 train_basil_complete.py`
5. **Wait:** Training completes (1-2 hours)
6. **Enjoy:** Use model for basil disease detection! 🌿

---

**Questions?** Read this file or check:
- `README_SETUP.md` - Setup details
- `DOCUMENTATION.md` - Technical reference
- `QUICK_START.md` - Quick reference

---

*Last updated: June 2026*
*For basil disease detection with YOLOv8*

