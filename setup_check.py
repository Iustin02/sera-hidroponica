#!/usr/bin/env python3
"""
🔧 SETUP SCRIPT - Detectează și configurează automat environment-ul
Rulați acest script după clone pentru a verifica setup-ul.
"""

import sys
import platform
import subprocess

def check_python():
    """Verifică versiunea Python."""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ⚠️  Recomandare: Python 3.8+")
        return False
    print("   ✅ OK")
    return True

def check_torch():
    """Verifică PyTorch și device-ul disponibil."""
    try:
        import torch
        print(f"\n🔥 PyTorch: {torch.__version__}")
        
        # CUDA
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   🎮 CUDA disponibil: {gpu}")
            print(f"   📊 VRAM: {vram:.1f} GB")
            print(f"   ✅ Device recomandat: cuda")
            return 'cuda'
        
        # MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   🍎 MPS disponibil (Apple Silicon)")
            print(f"   ✅ Device recomandat: mps")
            return 'mps'
        
        else:
            print(f"   💻 Doar CPU disponibil")
            print(f"   ⚠️  Antrenamentul va fi lent")
            return 'cpu'
            
    except ImportError:
        print("\n❌ PyTorch nu este instalat!")
        print("   Rulați: pip install torch torchvision")
        return None

def check_ultralytics():
    """Verifică Ultralytics (YOLOv8)."""
    try:
        import ultralytics
        print(f"\n📦 Ultralytics: {ultralytics.__version__}")
        print("   ✅ OK")
        return True
    except ImportError:
        print("\n❌ Ultralytics nu este instalat!")
        print("   Rulați: pip install ultralytics")
        return False

def check_dataset():
    """Verifică existența dataset-ului."""
    from pathlib import Path
    
    print("\n📁 Dataset:")
    
    paths_to_check = [
        ("data.yaml", "Configurație principală"),
        ("tomato_real/data.yaml", "Configurație tomato_real"),
        ("tomato_real/images/train", "Imagini train"),
        ("tomato_real/images/val", "Imagini validare"),
    ]
    
    found = False
    for path, desc in paths_to_check:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                count = len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png")))
                print(f"   ✅ {desc}: {path} ({count} imagini)")
            else:
                print(f"   ✅ {desc}: {path}")
            found = True
    
    if not found:
        print("   ⚠️  Nu am găsit dataset!")
        print("   Rulați: python prepare_dataset.py")
    
    return found

def print_recommendations(device):
    """Afișează recomandări pentru antrenament."""
    print("\n" + "=" * 50)
    print("📋 RECOMANDĂRI ANTRENAMENT")
    print("=" * 50)
    
    if device == 'cuda':
        print("""
Pentru NVIDIA GPU:
  python train.py --epochs 200 --model s --batch 32

Modele recomandate: YOLOv8s sau YOLOv8m
Timp estimat: 20-40 minute pentru 100 epoci
        """)
    elif device == 'mps':
        print("""
Pentru Apple Silicon:
  python train.py --epochs 100 --model n --batch 8

Modele recomandate: YOLOv8n
Timp estimat: 45-60 minute pentru 100 epoci
        """)
    else:
        print("""
Pentru CPU:
  python train.py --epochs 50 --model n --batch 4

⚠️  Antrenamentul pe CPU este foarte lent!
Recomandare: Folosiți Google Colab gratuit cu GPU.
        """)

def main():
    print("=" * 50)
    print("🔧 SERA HIDROPONICĂ - VERIFICARE SETUP")
    print("=" * 50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Arhitectură: {platform.machine()}")
    
    # Verificări
    check_python()
    device = check_torch()
    check_ultralytics()
    check_dataset()
    
    if device:
        print_recommendations(device)
    
    print("\n✅ Setup complet! Rulați: python train.py")

if __name__ == "__main__":
    main()
