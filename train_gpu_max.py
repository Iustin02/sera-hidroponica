#!/usr/bin/env python3
"""
🚀 ANTRENAMENT YOLOV8 - FORȚARE GPU LA MAXIM
RTX 3060 12GB + ~20GB RAM (optimizat)
"""

import os
import torch
import multiprocessing

def main():
    # Cache pe E:
    os.environ['TMPDIR'] = 'E:\\yolo_cache'
    os.environ['TEMP'] = 'E:\\yolo_cache'
    os.environ['TMP'] = 'E:\\yolo_cache'
    
    # Fix pentru shared memory pe Windows
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Sharing strategy pentru Windows
    torch.multiprocessing.set_sharing_strategy('file_system')

    print("=" * 70)
    print("🚀 ANTRENAMENT YOLOV8 - GPU MAXIM")
    print("=" * 70)

    # Verificăm GPU
    print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")

    # Setăm PyTorch pentru performanță maximă GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

    from ultralytics import YOLO

    # Încărcăm modelul
    print("\n📦 Încarc modelul YOLOv8n...")
    model = YOLO('yolov8n.pt')

    # Configurație pentru GPU MAXIM
    config = {
        'data': 'D:/sera-hidroponica/plantvillage_full/data.yaml',
        'epochs': 200,
        'batch': 32,              # Batch optimizat pentru workers + VRAM
        'imgsz': 640,
        'device': 0,              # GPU 0 explicit
        'workers': 4,             # Workers pentru viteză maximă
        'project': 'runs/detect',
        'name': 'plantvillage_200ep_gpu',
        'exist_ok': True,
        'patience': 100,
        'amp': True,              # Mixed precision - esențial
        'cache': False,           # Fără cache
        'save_period': 50,
        'verbose': True,
        'plots': True,
        
        # Augmentări
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        
        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
    }

    print(f"\n📋 Configurație:")
    print(f"   Batch size: {config['batch']}")
    print(f"   Workers: {config['workers']}")
    print(f"   AMP: {config['amp']} (Mixed Precision)")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Device: CUDA:0")
    print(f"   RAM Target: ~20GB")

    print("\n" + "=" * 70)
    print("🔥 START ANTRENAMENT - VERIFICĂ TASK MANAGER!")
    print("=" * 70 + "\n")

    # Antrenăm
    results = model.train(**config)

    print("\n✅ ANTRENAMENT FINALIZAT!")
    print(f"📁 Rezultate: runs/detect/plantvillage_1000ep_gpu/")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
