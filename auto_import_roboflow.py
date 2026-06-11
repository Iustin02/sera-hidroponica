#!/usr/bin/env python3
"""
Auto-import Roboflow dataset when downloaded
Monitors ~/Downloads for basil-health-detection ZIP file
"""
import os
import sys
import time
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("🌿 ROBOFLOW AUTO-IMPORT SYSTEM")
print("=" * 70)

# Configuration
downloads_dir = Path.home() / "Downloads"
project_dir = Path("/Users/iustiniuliandodita/sera-hidroponica")
basil_dataset_dir = project_dir / "basil_dataset"

# Expected ZIP filename pattern
zip_pattern = "basil-health-detection-*.zip"

print("\n📋 STEP 1: DOWNLOAD THE DATASET")
print("-" * 70)
print(f"""
1️⃣  OPEN this link in your browser:
   https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs

2️⃣  CLICK the green "Download" button (top right)

3️⃣  SELECT "YOLOv8" format from the dropdown

4️⃣  CLICK "Download" to save ZIP file to Downloads

5️⃣  The ZIP will appear as:
   ~/Downloads/basil-health-detection-*.zip
   (something like: basil-health-detection-1.zip)

""")

input("⏳ Press ENTER after you've downloaded the ZIP file...")

print("\n🔍 STEP 2: LOCATING DOWNLOADED FILE")
print("-" * 70)

# Find the ZIP file
zip_files = sorted(downloads_dir.glob("basil-health-detection-*.zip"), key=os.path.getmtime, reverse=True)

if not zip_files:
    print(f"❌ ZIP file not found in {downloads_dir}")
    print(f"\n📍 Expected: basil-health-detection-*.zip")
    print(f"\n⚠️  Make sure you:")
    print("   1. Downloaded from Roboflow link")
    print("   2. Selected YOLOv8 format")
    print("   3. Downloaded to ~/Downloads folder")
    sys.exit(1)

zip_file = zip_files[0]
print(f"✅ Found: {zip_file.name}")
print(f"   Size: {zip_file.stat().st_size / (1024*1024):.1f} MB")
print(f"   Time: {datetime.fromtimestamp(zip_file.stat().st_mtime)}")

print("\n📦 STEP 3: EXTRACTING ZIP FILE")
print("-" * 70)

# Extract to temporary directory
temp_dir = project_dir / "basil_dataset_temp"
if temp_dir.exists():
    print(f"🗑️  Removing old temp directory...")
    shutil.rmtree(temp_dir)

print(f"📂 Extracting to: {temp_dir}")

try:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print("✅ Extraction complete")
except Exception as e:
    print(f"❌ Extraction failed: {e}")
    sys.exit(1)

# Find the extracted directory
extracted_dirs = list(temp_dir.glob("basil-health-detection-*"))
if not extracted_dirs:
    extracted_dirs = list(temp_dir.glob("*/"))

if not extracted_dirs:
    print("❌ Could not find extracted dataset directory")
    sys.exit(1)

dataset_root = extracted_dirs[0]
print(f"📁 Dataset root: {dataset_root.name}")

print("\n📂 STEP 4: PREPARING DATASET STRUCTURE")
print("-" * 70)

# Check for images and labels
source_images_dir = dataset_root / "images"
source_labels_dir = dataset_root / "labels"

if not source_images_dir.exists():
    print(f"❌ No images/ directory found in {dataset_root}")
    sys.exit(1)

if not source_labels_dir.exists():
    print(f"❌ No labels/ directory found in {dataset_root}")
    sys.exit(1)

# Count images
train_images = len(list((source_images_dir / "train").glob("*"))) if (source_images_dir / "train").exists() else 0
val_images = len(list((source_images_dir / "val").glob("*"))) if (source_images_dir / "val").exists() else 0
test_images = len(list((source_images_dir / "test").glob("*"))) if (source_images_dir / "test").exists() else 0
total_images = train_images + val_images + test_images

print(f"📊 Dataset contents:")
print(f"   Train: {train_images} images")
print(f"   Val:   {val_images} images")
print(f"   Test:  {test_images} images")
print(f"   Total: {total_images} images")

if total_images == 0:
    print(f"\n❌ No images found in dataset!")
    sys.exit(1)

print("\n🔄 STEP 5: COPYING TO PROJECT")
print("-" * 70)

# Clear old images and labels
print("🗑️  Clearing old sample data...")
for split in ["train", "val", "test"]:
    images_path = basil_dataset_dir / "images" / split
    if images_path.exists():
        shutil.rmtree(images_path)
    images_path.mkdir(parents=True, exist_ok=True)
    
    labels_path = basil_dataset_dir / "labels" / split
    if labels_path.exists():
        shutil.rmtree(labels_path)
    labels_path.mkdir(parents=True, exist_ok=True)

# Copy images
print("📋 Copying images...")
for split in ["train", "val", "test"]:
    src = source_images_dir / split
    dst = basil_dataset_dir / "images" / split
    
    if src.exists():
        for img_file in src.glob("*"):
            shutil.copy2(img_file, dst / img_file.name)
        count = len(list(dst.glob("*")))
        print(f"   ✓ {split}/: {count} images")

# Copy labels
print("📋 Copying labels...")
for split in ["train", "val", "test"]:
    src = source_labels_dir / split
    dst = basil_dataset_dir / "labels" / split
    
    if src.exists():
        for label_file in src.glob("*.txt"):
            shutil.copy2(label_file, dst / label_file.name)
        count = len(list(dst.glob("*.txt")))
        print(f"   ✓ {split}/: {count} labels")

print("\n✅ STEP 6: VERIFYING DATASET")
print("-" * 70)

# Run verification
import subprocess

result = subprocess.run([sys.executable, "verify_dataset.py"], capture_output=True, text=True)
print(result.stdout)

if result.returncode != 0:
    print(result.stderr)
    print("⚠️  Verification had issues")
else:
    print("✅ Dataset verification passed!")

print("\n🗑️  STEP 7: CLEANUP")
print("-" * 70)

# Remove temporary directory
print(f"Removing temporary files...")
shutil.rmtree(temp_dir)
print("✅ Cleanup complete")

print("\n" + "=" * 70)
print("🎉 DATASET IMPORT SUCCESSFUL!")
print("=" * 70)

print(f"""
✅ Dataset is ready in: {basil_dataset_dir}

📊 Summary:
   Train images: {train_images}
   Val images:   {val_images}
   Test images:  {test_images}
   Total:        {total_images}

🚀 NEXT STEP: Train the model

   Run this command:
   python3 train_basil_complete.py

   Training will take ~1-2 hours on CPU
   Monitor progress with: open runs/detect/train/
""")

print("=" * 70)

