#!/usr/bin/env python3
"""
🔧 SETUP SCRIPT - Detectează și configurează automat environment-ul
Rulați acest script după clone pentru a verifica setup-ul.
"""

import sys
import platform
import subprocess

# Evită UnicodeEncodeError pe Windows (mai ales când output-ul e piped)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

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
    """Verifică existența unui dataset YOLO (local)."""
    from pathlib import Path

    print("\n📁 Dataset:")

    # Dataset-ul iNaturalist (busuioc/mentă) este ținut local și ignorat din git.
    default_inat = Path("inat_datasets/inat_basil_mint_20k/data.yaml")

    paths_to_check = [
        (default_inat, "Configurație iNaturalist (busuioc/mentă)"),
        (Path("inat_datasets"), "Folder iNaturalist (local)"),
    ]

    found = False
    for p, desc in paths_to_check:
        if p.exists():
            if p.is_dir():
                n = len(list(p.rglob("*.jpg"))) + len(list(p.rglob("*.png")))
                print(f"   ✅ {desc}: {p} ({n} imagini)")
            else:
                print(f"   ✅ {desc}: {p}")
            found = True

    if not found:
        print("   ⚠️  Nu am găsit niciun dataset local.")
        print("   Pentru busuioc/mentă (iNaturalist), rulează:")
        print("     python download_inat_yolo.py --out inat_datasets\\inat_basil_mint_20k --per-class 10000 --val-per-class 1000")
        print("   Sau rulează train cu --data către propriul data.yaml.")

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
