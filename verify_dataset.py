#!/usr/bin/env python3
"""
Verify Basil dataset structure and prepare for training
"""

import os
from pathlib import Path

def verify_dataset():
    """Check if dataset is properly structured"""
    
    print("=" * 60)
    print("🔍 DATASET VERIFICATION")
    print("=" * 60)
    
    dataset_path = Path("basil_dataset")
    
    # Check main folders
    print("\n📁 Checking directory structure...")
    required_dirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {dir_path}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n⚠️  Some directories missing!")
        return False
    
    # Check images
    print("\n📊 Image counts...")
    stats = {}
    for split in ["train", "val", "test"]:
        img_dir = dataset_path / "images" / split
        count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        stats[split] = count
        print(f"   {split}: {count} images")
    
    # Check labels
    print("\n📝 Label counts...")
    for split in ["train", "val", "test"]:
        lbl_dir = dataset_path / "labels" / split
        count = len(list(lbl_dir.glob("*.txt")))
        expected = stats[split]
        match = "✅" if count == expected else "❌"
        print(f"   {match} {split}: {count} labels (expected {expected})")
    
    # Check data.yaml
    print("\n📋 Configuration...")
    yaml_file = dataset_path / "data.yaml"
    if yaml_file.exists():
        print(f"   ✅ data.yaml exists")
        with open(yaml_file) as f:
            content = f.read()
            if "nc: 5" in content and "healthy" in content:
                print("   ✅ Classes configured correctly")
            else:
                print("   ⚠️  Classes might not be configured")
    else:
        print(f"   ❌ data.yaml missing!")
    
    total_images = sum(stats.values())
    
    print("\n" + "=" * 60)
    if total_images > 0:
        print(f"✅ DATASET READY FOR TRAINING!")
        print(f"   Total images: {total_images}")
        return True
    else:
        print("⚠️  Dataset is empty! Import data from Roboflow")
        return False

if __name__ == "__main__":
    ready = verify_dataset()
    if not ready:
        print("\n📥 Next: Download data from Roboflow")
        print("   https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs")

