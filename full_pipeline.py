#!/usr/bin/env python3
"""
Complete Basil YOLOv8 Pipeline:
1. Wait for Roboflow download
2. Auto-import dataset
3. Auto-train model
"""
import os
import sys
import time
import shutil
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

def print_header(text):
    print("\n" + "=" * 70)
    print(f"🌿 {text}")
    print("=" * 70)

def print_step(num, text):
    print(f"\n{'='*70}")
    print(f"STEP {num}: {text}")
    print(f"{'='*70}\n")

# Configuration
downloads_dir = Path.home() / "Downloads"
project_dir = Path("/Users/iustiniuliandodita/sera-hidroponica")
basil_dataset_dir = project_dir / "basil_dataset"

print_header("BASIL YOLOv8 COMPLETE PIPELINE")

print(f"""
This script will:
1. Wait for Roboflow dataset download
2. Import dataset automatically
3. Verify dataset integrity
4. Train YOLOv8 model (50 epochs)
5. Show training results

Estimated time: 1.5 - 2.5 hours

Let's get started! 🚀
""")

# ============================================================================
# STEP 1: WAIT FOR DOWNLOAD
# ============================================================================
print_step(1, "DOWNLOAD ROBOFLOW DATASET")

print("""
INSTRUCTIONS:
1. Open in browser: https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs
2. Click green "Download" button (top right)
3. Select "YOLOv8" format
4. Click "Download" to save to ~/Downloads/

The ZIP file will be named: basil-health-detection-*.zip

Waiting for download... (press Ctrl+C to cancel)
""")

# Find ZIP file
zip_file = None
check_interval = 5  # seconds
max_wait_time = 3600  # 1 hour
elapsed = 0

while not zip_file and elapsed < max_wait_time:
    zip_files = sorted(downloads_dir.glob("basil-health-detection-*.zip"), key=os.path.getmtime, reverse=True)
    
    if zip_files:
        zip_file = zip_files[0]
        print(f"\n✅ Found: {zip_file.name}")
        print(f"   Size: {zip_file.stat().st_size / (1024*1024):.1f} MB")
        break
    
    # Show progress
    print(f"⏳ Waiting... ({elapsed}s elapsed)", end="\r")
    time.sleep(check_interval)
    elapsed += check_interval

if not zip_file:
    print(f"\n❌ Download not found after {max_wait_time} seconds")
    print(f"Check: ~/Downloads/basil-health-detection-*.zip")
    sys.exit(1)

print(f"✅ File found and ready for processing")

# ============================================================================
# STEP 2: EXTRACT AND IMPORT
# ============================================================================
print_step(2, "EXTRACT AND IMPORT DATASET")

# Extract
temp_dir = project_dir / "basil_dataset_temp"
if temp_dir.exists():
    print("🗑️  Removing old temp directory...")
    shutil.rmtree(temp_dir)

print(f"📂 Extracting {zip_file.name}...")

try:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print("✅ Extraction complete")
except Exception as e:
    print(f"❌ Extraction failed: {e}")
    sys.exit(1)

# Find extracted directory
extracted_dirs = list(temp_dir.glob("basil-health-detection-*"))
if not extracted_dirs:
    extracted_dirs = list(temp_dir.glob("*/"))

if not extracted_dirs:
    print("❌ Could not find extracted dataset")
    sys.exit(1)

dataset_root = extracted_dirs[0]
print(f"✅ Located: {dataset_root.name}")

# Check structure
source_images_dir = dataset_root / "images"
source_labels_dir = dataset_root / "labels"

if not source_images_dir.exists() or not source_labels_dir.exists():
    print("❌ Invalid dataset structure (missing images or labels)")
    sys.exit(1)

# Count images before copy
train_images = len(list((source_images_dir / "train").glob("*"))) if (source_images_dir / "train").exists() else 0
val_images = len(list((source_images_dir / "val").glob("*"))) if (source_images_dir / "val").exists() else 0
test_images = len(list((source_images_dir / "test").glob("*"))) if (source_images_dir / "test").exists() else 0
total_images = train_images + val_images + test_images

print(f"\n📊 Dataset contents:")
print(f"   Train: {train_images}")
print(f"   Val:   {val_images}")
print(f"   Test:  {test_images}")
print(f"   Total: {total_images}")

if total_images == 0:
    print("❌ No images found in dataset")
    sys.exit(1)

# Clear old data
print("\n🗑️  Clearing old sample data...")
for split in ["train", "val", "test"]:
    for img_path in (basil_dataset_dir / "images" / split).glob("*"):
        img_path.unlink()
    for lbl_path in (basil_dataset_dir / "labels" / split).glob("*.txt"):
        lbl_path.unlink()

# Copy data
print("📋 Copying dataset...")
for split in ["train", "val", "test"]:
    src = source_images_dir / split
    dst = basil_dataset_dir / "images" / split
    
    if src.exists():
        for img_file in src.glob("*"):
            shutil.copy2(img_file, dst / img_file.name)

for split in ["train", "val", "test"]:
    src = source_labels_dir / split
    dst = basil_dataset_dir / "labels" / split
    
    if src.exists():
        for label_file in src.glob("*.txt"):
            shutil.copy2(label_file, dst / label_file.name)

print("✅ Import complete")

# ============================================================================
# STEP 3: VERIFY DATASET
# ============================================================================
print_step(3, "VERIFY DATASET INTEGRITY")

result = subprocess.run([sys.executable, "verify_dataset.py"], capture_output=True, text=True)
print(result.stdout)

if result.returncode != 0:
    print("⚠️  Verification issues detected")
    print(result.stderr)
else:
    print("✅ Verification passed!")

# ============================================================================
# STEP 4: CLEANUP TEMP FILES
# ============================================================================
print_step(4, "CLEANUP TEMPORARY FILES")

print("🗑️  Removing temporary files...")
if temp_dir.exists():
    shutil.rmtree(temp_dir)
print("✅ Cleanup complete")

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================
print_step(5, "TRAIN YOLOV8 MODEL")

print(f"""
🚀 Starting training with {total_images} images

Configuration:
   Model: YOLOv8 nano
   Epochs: 50
   Batch: 16
   Images: 640x640
   Device: CPU (GPU auto-detected if available)

Training duration: 1-2 hours on CPU

Starting training...
""")

input("Press ENTER to start training (or Ctrl+C to cancel)...")

# Run training
result = subprocess.run([sys.executable, "train_basil_complete.py"])

if result.returncode == 0:
    print_header("TRAINING COMPLETE! ✅")
    print(f"""
✅ Model successfully trained!

📊 Results location:
   runs/detect/train/

📁 Key files:
   • weights/best.pt - Best model (use for inference)
   • weights/last.pt - Last epoch model
   • results.png - Training plots
   • confusion_matrix.png - Per-class accuracy

📖 Next steps:
   1. View results: open runs/detect/train/
   2. Use model: Load best.pt for inference
   3. Test predictions: python3 inference.py

🎉 Your basil disease detection model is ready!
""")
else:
    print_header("TRAINING ENDED")
    print("⚠️  Training process ended (may have been stopped manually)")

print("\n" + "=" * 70)
print("Pipeline script completed!")
print("=" * 70)

