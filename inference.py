#!/usr/bin/env python3
"""
Basil Disease Detection - Inference Script
Uses trained YOLOv8 model to detect diseases in basil images
"""

import sys
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--break-system-packages", "-q"])
    from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Basil Disease Detection")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", 
                       help="Path to model (default: best trained model)")
    parser.add_argument("--source", type=str, default="basil_dataset/images/val/",
                       help="Image or directory to analyze")
    parser.add_argument("--conf", type=float, default=0.5, 
                       help="Confidence threshold (0-1)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu or 0 for GPU)")
    
    args = parser.parse_args()
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {args.model}")
        print("\n📍 Have you trained the model yet?")
        print("   Run: python3 full_pipeline.py")
        sys.exit(1)
    
    # Check source exists
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"❌ Source not found: {args.source}")
        sys.exit(1)
    
    print(f"🌿 BASIL DISEASE DETECTION")
    print(f"{'='*70}")
    print(f"\n📦 Loading model: {model_path.name}")
    model = YOLO(str(model_path))
    
    print(f"🔍 Analyzing images from: {args.source}")
    print(f"   Confidence threshold: {args.conf}")
    print(f"   Device: {args.device}\n")
    
    # Run inference
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        device=args.device,
        verbose=False
    )
    
    # Process results
    disease_classes = {
        0: "Healthy 🌿",
        1: "Downy Mildew 🌫️",
        2: "Bacterial Leaf Spot 🔴",
        3: "Leaf Miner Damage 🪲",
        4: "General Disease ⚠️"
    }
    
    total_detections = 0
    disease_counts = {i: 0 for i in range(5)}
    
    print("📊 RESULTS:")
    print(f"{'='*70}\n")
    
    for result in results:
        image_path = result.path
        boxes = result.boxes
        
        if len(boxes) == 0:
            print(f"📷 {Path(image_path).name}: No diseases detected ✅")
        else:
            print(f"📷 {Path(image_path).name}:")
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = disease_classes.get(class_id, f"Class {class_id}")
                
                print(f"   • {class_name} - Confidence: {confidence:.1%}")
                disease_counts[class_id] += 1
                total_detections += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📈 SUMMARY:")
    print(f"{'='*70}\n")
    print(f"Total detections: {total_detections}")
    print(f"\nDetections by disease:")
    
    for class_id, count in disease_counts.items():
        class_name = disease_classes[class_id]
        print(f"   {class_name}: {count}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()

