#!/usr/bin/env python3
"""
Download and prepare Basil dataset from PlantVillage
Converts classification format to YOLO detection format
"""

import os
import sys
from pathlib import Path
import shutil
import random

def download_plantvillage_basil():
    """Download PlantVillage dataset (basil only)"""
    print("📥 Downloading PlantVillage Basil dataset...")
    
    # Clone if not exists
    if not Path("PlantVillage-Dataset").exists():
        print("   Cloning repository...")
        os.system("git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git 2>/dev/null")
    
    pv_path = Path("PlantVillage-Dataset/raw/color")
    
    if not pv_path.exists():
        print("❌ PlantVillage dataset not found!")
        return False
    
    # Find basil classes
    basil_classes = {}
    for class_dir in pv_path.iterdir():
        if class_dir.is_dir() and "Basil" in class_dir.name:
            images = list(class_dir.glob("*.jpg"))
            basil_classes[class_dir.name] = images
            print(f"   ✅ {class_dir.name}: {len(images)} images")
    
    if not basil_classes:
        print("❌ No basil classes found!")
        return False
    
    return basil_classes

def map_classes():
    """Map PlantVillage class names to YOLO format"""
    return {
        "Basil___healthy": ("healthy", 0),
        "Basil___Downy_mildew": ("downy_mildew", 1),
        "Basil___Leaf_Spot": ("leaf_spot", 2),
        "Basil___Powdery_Mildew": ("powdery_mildew", 3),
    }

def create_yolo_label(class_id):
    """Create YOLO format label (center-based bbox normalized)"""
    # For isolated leaf images, use full frame as bounding box
    return f"{class_id} 0.5 0.5 0.9 0.9\n"

def organize_dataset(basil_classes):
    """Organize images into train/val/test split"""
    print("\n📊 Organizing dataset...")
    
    class_mapping = map_classes()
    dataset_path = Path("basil_dataset")
    
    splits = {"train": 0.7, "val": 0.2, "test": 0.1}
    
    total_images = 0
    
    for pv_class, images in basil_classes.items():
        if pv_class not in class_mapping:
            print(f"   ⏭️  Skipping: {pv_class}")
            continue
        
        yolo_class, class_id = class_mapping[pv_class]
        
        # Shuffle and split
        random.shuffle(images)
        train_count = int(len(images) * splits["train"])
        val_count = int(len(images) * splits["val"])
        
        splits_dict = {
            "train": images[:train_count],
            "val": images[train_count:train_count+val_count],
            "test": images[train_count+val_count:]
        }
        
        # Copy files
        for split, split_images in splits_dict.items():
            for img_file in split_images:
                # Copy image
                img_dest = dataset_path / "images" / split / f"{pv_class}_{img_file.name}"
                shutil.copy2(img_file, img_dest)
                
                # Create label
                label_name = img_dest.stem + ".txt"
                label_dest = dataset_path / "labels" / split / label_name
                label_dest.write_text(create_yolo_label(class_id))
                
                total_images += 1
        
        print(f"   ✅ {yolo_class}: {len(images)} images")
    
    print(f"\n✅ Total images organized: {total_images}")
    
    # Print split stats
    for split in ["train", "val", "test"]:
        count = len(list((dataset_path / "images" / split).glob("*.*")))
        print(f"   {split}: {count}")

if __name__ == "__main__":
    print("=" * 60)
    print("🌿 BASIL DATASET PREPARATION")
    print("=" * 60)
    
    # Download
    basil_classes = download_plantvillage_basil()
    
    if basil_classes:
        # Organize
        organize_dataset(basil_classes)
        print("\n✅ Dataset ready at: basil_dataset/")
        print("   Configure in: data.yaml")
        print("   Train with: yolo detect train data=basil_dataset/data.yaml")
    else:
        print("\n⚠️  Could not prepare dataset")
        sys.exit(1)

