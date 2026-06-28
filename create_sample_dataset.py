#!/usr/bin/env python3
"""
Create sample Basil images for testing YOLOv8 training pipeline
(Simulates real data for testing before importing Roboflow dataset)
"""

import os
import random
from pathlib import Path
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    print("Installing pillow...")
    os.system("pip install pillow --break-system-packages -q")
    from PIL import Image, ImageDraw, ImageFilter

def create_sample_images():
    """Generate sample basil leaf images for each class"""
    
    print("=" * 60)
    print("🌿 CREATING SAMPLE DATASET FOR TESTING")
    print("=" * 60)
    
    dataset_path = Path("basil_dataset")
    classes = {
        0: "healthy",
        1: "downy_mildew",
        2: "bacterial_leaf_spot",
        3: "leaf_miner_damage",
        4: "general_disease"
    }
    
    # Create 50 sample images total (split across classes and splits)
    samples_per_class = 10  # Total
    
    print(f"\n🎨 Generating {samples_per_class * len(classes)} sample images...")
    
    for class_id, class_name in classes.items():
        print(f"\n   Creating {class_name}...")
        
        for i in range(samples_per_class):
            # Decide split
            rand = random.random()
            if rand < 0.7:
                split = "train"
            elif rand < 0.9:
                split = "val"
            else:
                split = "test"
            
            # Create image
            img = create_basil_leaf_image(class_id, class_name)
            
            # Save image
            img_name = f"basil_{class_name}_{i:04d}.jpg"
            img_path = dataset_path / "images" / split / img_name
            img.save(img_path)
            
            # Create YOLO label
            # Normalized bbox: x_center, y_center, width, height (0-1)
            # For full leaf fill: 0.5, 0.5, 0.8, 0.9
            label = f"{class_id} 0.5 0.5 0.8 0.9\n"
            label_path = dataset_path / "labels" / split / (Path(img_name).stem + ".txt")
            label_path.write_text(label)
        
        print(f"       ✅ Created {samples_per_class} {class_name} samples")
    
    print("\n✅ Sample dataset created!")
    
    # Verify
    total = 0
    for split in ["train", "val", "test"]:
        count = len(list((dataset_path / "images" / split).glob("*.jpg")))
        total += count
        print(f"   {split}: {count} images")
    
    print(f"   Total: {total} images")

def create_basil_leaf_image(class_id, class_name, size=640):
    """Generate a synthetic basil leaf image based on disease type"""
    
    # Base image with green background
    img = Image.new("RGB", (size, size), color=(34, 139, 34))  # Forest green
    
    # Add leaf shape - ellipse in center
    draw = ImageDraw.Draw(img, "RGBA")
    
    # Leaf center and size
    center_x, center_y = size // 2, size // 2
    leaf_width = size // 2
    leaf_height = int(size * 0.65)
    
    # Draw leaf shape
    left = center_x - leaf_width // 2
    top = center_y - leaf_height // 2
    right = center_x + leaf_width // 2
    bottom = center_y + leaf_height // 2
    
    # Base leaf color - green variations
    if class_id == 0:  # Healthy
        leaf_color = (76, 175, 80)  # Light green
    else:
        leaf_color = (66, 165, 60)  # Slightly darker
    
    draw.ellipse([left, top, right, bottom], fill=leaf_color)
    
    # Add disease patterns
    if class_id == 1:  # Downy mildew - white/gray spots
        for _ in range(15):
            x = random.randint(left + 50, right - 50)
            y = random.randint(top + 50, bottom - 50)
            r = random.randint(10, 30)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(200, 200, 200, 150))
    
    elif class_id == 2:  # Bacterial leaf spot - dark brown spots
        for _ in range(20):
            x = random.randint(left + 50, right - 50)
            y = random.randint(top + 50, bottom - 50)
            r = random.randint(5, 15)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(139, 69, 19))
    
    elif class_id == 3:  # Leaf miner - light lines/tunnels
        for _ in range(8):
            x1 = random.randint(left + 50, right - 50)
            y1 = random.randint(top + 50, bottom - 50)
            x2 = x1 + random.randint(-80, 80)
            y2 = y1 + random.randint(-80, 80)
            draw.line([x1, y1, x2, y2], fill=(255, 255, 180), width=3)
    
    elif class_id == 4:  # General disease - mixed pattern
        for _ in range(12):
            x = random.randint(left + 50, right - 50)
            y = random.randint(top + 50, bottom - 50)
            r = random.randint(8, 20)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(200, 100, 100, 120))
    
    # Add vein texture
    draw.line([center_x, top, center_x, bottom], fill=(40, 100, 40), width=2)
    for i in range(5):
        offset = i * 30
        draw.line([center_x - offset, top + 100, center_x - offset - 50, bottom - 100], 
                 fill=(50, 120, 50), width=1)
        draw.line([center_x + offset, top + 100, center_x + offset + 50, bottom - 100], 
                 fill=(50, 120, 50), width=1)
    
    # Add text label
    try:
        draw.text((10, 10), f"Class: {class_name}", fill=(255, 255, 255))
    except:
        pass  # Ignore if font fails
    
    # Add some noise
    np_img = np.array(img)
    noise = np.random.normal(0, 5, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(np_img)
    
    return img

if __name__ == "__main__":
    create_sample_images()

