#!/usr/bin/env python3
"""
Download basil dataset alternative sources
"""
import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path

print("=" * 70)
print("🌿 BASIL DATASET DOWNLOAD - ALTERNATIVE SOURCES")
print("=" * 70)

# Try PlantVillage dataset (public)
print("\n🔍 Attempting download from PlantVillage dataset...")
print("📍 This is a comprehensive plant disease dataset")

try:
    print("\n⏬ Downloading PlantVillage dataset (large: ~750 MB)...")
    
    url = "https://github.com/spMohanty/PlantVillage-Dataset/releases/download/v1.0/PlantVillage-0-raw.tar.gz"
    
    dataset_file = Path("plantvillage_raw.tar.gz")
    
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (compatible; DataBot/1.0)'}
    )
    
    print(f"   URL: {url[:60]}...")
    
    # This might fail too, but let's try
    urllib.request.urlretrieve(url, dataset_file)
    
    print(f"✅ Downloaded: {dataset_file}")
    
except Exception as e:
    print(f"❌ PlantVillage download failed: {str(e)[:100]}")

print("\n" + "=" * 70)
print("ℹ️  SOLUTIONS:")
print("=" * 70)

print("\n✅ OPTION 1: Manual Roboflow Download (RECOMMENDED)")
print("   1. Go to: https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs")
print("   2. Click green 'Download' button (top right)")
print("   3. Select 'YOLOv8' format")
print("   4. Download as ZIP")
print("   5. Extract: unzip ~/Downloads/basil-health-detection-*.zip -d basil_dataset_roboflow/")
print("   6. Copy: cp -r basil_dataset_roboflow/images/* basil_dataset/images/")
print("   7. Copy: cp -r basil_dataset_roboflow/labels/* basil_dataset/labels/")
print("   8. Verify: python3 verify_dataset.py")
print("   9. Train: python3 train_basil_complete.py")

print("\n✅ OPTION 2: Use Sample Data for Testing")
print("   ✓ We already have 50 synthetic basil images ready")
print("   ✓ Run: python3 train_quick.py")
print("   ✓ This proves the pipeline works")
print("   ✓ Then download real data and import")

print("\n✅ OPTION 3: Create Custom Dataset")
print("   1. Take photos of your basil plants")
print("   2. Use Roboflow Studio to annotate")
print("   3. Export as YOLOv8 format")
print("   4. Place in basil_dataset/")

print("\n" + "=" * 70)
print("📊 Current Dataset Status:")
print("=" * 70)

print("\n✅ Sample Data Ready:")
train_count = len(list(Path("basil_dataset/images/train").glob("*")))
val_count = len(list(Path("basil_dataset/images/val").glob("*")))
test_count = len(list(Path("basil_dataset/images/test").glob("*")))

print(f"   Train: {train_count} images")
print(f"   Val:   {val_count} images")
print(f"   Test:  {test_count} images")
print(f"   Total: {train_count + val_count + test_count} images")

print("\n⏳ Real Dataset (Roboflow): 1,670 images - AWAITING DOWNLOAD")

print("\n" + "=" * 70)
print("🚀 RECOMMENDED NEXT STEP:")
print("=" * 70)
print("""
1. Visit Roboflow link (Option 1 above)
2. Download the ZIP file manually
3. Extract to project
4. Run: python3 train_basil_complete.py

OR

1. Test with sample data first:
   python3 train_quick.py
2. Then download real data and train with it
""")

print("=" * 70)

