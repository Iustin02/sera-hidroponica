#!/usr/bin/env python3
"""
Script de test pentru verificarea suportului MPS (Metal Performance Shaders)
pe MacBook Pro M1 Pro - Proiect Seră Hidroponică Smart
"""

import torch
import platform

def test_mps_support():
    print("=" * 60)
    print("🍎 Test PyTorch MPS pentru Apple Silicon")
    print("=" * 60)
    
    # Informații sistem
    print(f"\n📌 Sistem: {platform.system()} {platform.machine()}")
    print(f"📌 Python: {platform.python_version()}")
    print(f"📌 PyTorch: {torch.__version__}")
    
    # Verificare MPS
    print("\n" + "-" * 60)
    print("🔍 Verificare suport MPS:")
    print("-" * 60)
    
    mps_built = torch.backends.mps.is_built()
    print(f"   ├── MPS compilat în PyTorch: {'✅ Da' if mps_built else '❌ Nu'}")
    
    if mps_built:
        mps_available = torch.backends.mps.is_available()
        print(f"   └── MPS disponibil pe acest device: {'✅ Da' if mps_available else '❌ Nu'}")
        
        if mps_available:
            # Test practic cu tensor
            print("\n" + "-" * 60)
            print("🧪 Test practic - Operație pe GPU M1:")
            print("-" * 60)
            
            # Creăm un tensor pe MPS
            device = torch.device("mps")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # Înmulțire matricială pe GPU
            z = torch.matmul(x, y)
            
            print(f"   ├── Tensor creat pe: {x.device}")
            print(f"   ├── Dimensiune tensor: {x.shape}")
            print(f"   └── Înmulțire matricială: ✅ Succes!")
            
            # Verificare pentru YOLOv8
            print("\n" + "-" * 60)
            print("🚀 Recomandare pentru antrenament YOLOv8:")
            print("-" * 60)
            print("   Folosește device='mps' în comenzile YOLO:")
            print("   >>> from ultralytics import YOLO")
            print("   >>> model = YOLO('yolov8n.pt')")
            print("   >>> model.train(data='dataset.yaml', device='mps')")
            
            return True
    
    print("\n⚠️  MPS nu este disponibil. Verifică instalarea PyTorch.")
    return False


if __name__ == "__main__":
    success = test_mps_support()
    print("\n" + "=" * 60)
    if success:
        print("✅ GPU M1 este gata pentru antrenament YOLOv8!")
    else:
        print("❌ Probleme cu suportul MPS")
    print("=" * 60 + "\n")
