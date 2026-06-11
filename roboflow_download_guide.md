# 📥 ROBOFLOW DOWNLOAD GUIDE - BASIL DATASET

## 🔗 Step 1: Accesează Roboflow
```
https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs
```

## 📥 Step 2: Download
1. Click butonul verde **"Download"**
2. Selectează **"YOLOv8"** format
3. Copy link-ul generat (or click direct download)

## 📦 Step 3: Extract
```bash
# După ce descărci basil-health-detection-1.zip (sau alt număr versiune)
unzip basil-health-detection-*.zip -d roboflow_download/
ls roboflow_download/
```

## 📋 Step 4: Organizare
Structura downloadului va fi:
```
roboflow_download/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml (keep this)
```

## ✅ Step 5: Copy in Project
```bash
# Copiază imagini
cp -r roboflow_download/images/* basil_dataset/images/

# Copiază labels  
cp -r roboflow_download/labels/* basil_dataset/labels/

# Verify
ls basil_dataset/images/train/ | head -5
ls basil_dataset/labels/train/ | head -5
```

## 🎯 After Import
Run training:
```bash
python3 train_basil_complete.py
```
