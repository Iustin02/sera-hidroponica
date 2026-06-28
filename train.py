#!/usr/bin/env python3
"""
🌿 SERA HIDROPONICĂ SMART - ANTRENAMENT YOLOV8
Script hibrid: funcționează pe CUDA (NVIDIA), MPS (Apple Silicon) sau CPU.

Acest repo nu mai include training/dataset pentru roșii. Exemplul curent este
pentru busuioc/mentă (iNaturalist) sau orice dataset YOLO furnizat prin --data.
"""

import os
import sys
import torch
import platform
from pathlib import Path
from datetime import datetime

import yaml

# Evită UnicodeEncodeError pe Windows (mai ales când output-ul e piped)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Setări pentru a evita probleme cu matplotlib
os.environ['MPLBACKEND'] = 'Agg'

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _get_train_images_dir(data_yaml: str) -> Path:
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding='utf-8'))
    root = Path(cfg.get('path', Path(data_yaml).parent))
    train = cfg.get('train')
    if not train:
        return root

    p = Path(train)
    return p if p.is_absolute() else (root / p)


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _resolve_cache_mode(cache_arg: str, device: str, data_yaml: str):
    """Mapează CLI cache -> parametrul ultralytics `cache`.

    Ultralytics acceptă: False | True('ram') | 'disk'.
    """
    cache_arg = (cache_arg or 'auto').lower()

    if cache_arg == 'false':
        return False
    if cache_arg == 'disk':
        return 'disk'
    if cache_arg == 'ram':
        return True

    # auto
    if device != 'cuda':
        return False

    n = _count_images(_get_train_images_dir(data_yaml))
    # Pe dataset-uri mari evităm RAM cache (poate crăpa); disk cache e mai sigur.
    return 'disk' if n >= 8000 else True

def detect_device():
    """
    Detectează automat cel mai bun device disponibil.
    Prioritate: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 NVIDIA GPU detectat: {gpu_name}")
        print(f"   VRAM: {vram:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"🍎 Apple Silicon detectat (MPS)")
        print(f"   Chip: {platform.processor()}")
    else:
        device = 'cpu'
        print(f"💻 Folosesc CPU")
        print(f"   ⚠️  Antrenamentul va fi lent!")
    
    return device


def get_optimal_batch_size(device):
    """
    Returnează batch size optim în funcție de device și VRAM disponibil.
    """
    if device == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram >= 12:      # RTX 3060 12GB, RTX 3080, etc.
            return 32
        elif vram >= 8:     # RTX 3070, RTX 2080, etc.
            return 16
        elif vram >= 6:     # RTX 3060 6GB, GTX 1660, etc.
            return 8
        else:
            return 4
    elif device == 'mps':
        return 8  # Apple Silicon - conservator
    else:
        return 4  # CPU


def train(
    data_yaml: str = None,
    epochs: int = 100,
    model_size: str = 'n',  # n=nano, s=small, m=medium, l=large, x=xlarge
    batch_size: int = None,
    imgsz: int = 640,
    project_name: str = 'herbs_yolo',
    resume: bool = False,
    cache: str = 'auto',
):
    """
    Antrenează modelul YOLOv8 pentru detecția bolilor.
    
    Args:
        data_yaml: Calea către fișierul data.yaml
        epochs: Numărul de epoci
        model_size: Dimensiunea modelului (n/s/m/l/x)
        batch_size: Dimensiunea batch-ului (auto dacă None)
        imgsz: Dimensiunea imaginilor
        project_name: Numele proiectului pentru salvare
        resume: Continuă antrenamentul anterior
    """
    
    print("=" * 60)
    print("🌿 SERA HIDROPONICĂ - ANTRENAMENT YOLOV8")
    print("=" * 60)
    print(f"   Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print()
    
    # Detectează device-ul
    device = detect_device()
    
    # Batch size optim
    if batch_size is None:
        batch_size = get_optimal_batch_size(device)
    
    print(f"\n📋 Configurație antrenament:")
    print(f"   Device: {device}")
    print(f"   Model: YOLOv8{model_size}")
    print(f"   Epoci: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {imgsz}x{imgsz}")
    print(f"   Cache: {cache}")
    
    # Calea către data.yaml
    if data_yaml is None:
        # Caută automat
        possible_paths = []

        # 1) iNaturalist herbs dataset (default)
        default_inat = Path("inat_datasets/inat_basil_mint_20k/data.yaml")
        possible_paths.append(default_inat)

        # 2) Orice alt dataset local în inat_datasets/**/data.yaml
        inat_root = Path("inat_datasets")
        if inat_root.exists():
            possible_paths += sorted(inat_root.rglob("data.yaml"))

        # 3) Fallback pentru dataset-uri custom în root
        possible_paths.append(Path("data.yaml"))

        for p in possible_paths:
            if p.exists():
                data_yaml = str(p)
                break
        
        if data_yaml is None:
            print("\n❌ Nu am găsit data.yaml!")
            print("   Creează un dataset sau specifică calea manual.")
            sys.exit(1)
    
    print(f"   Dataset: {data_yaml}")
    
    # Import ultralytics
    from ultralytics import YOLO
    
    # Încarcă modelul
    model_name = f'yolov8{model_size}.pt'
    print(f"\n📦 Încarc modelul {model_name}...")
    model = YOLO(model_name)
    
    # Configurație workers
    workers = 8 if device == 'cuda' else 0

    cache_mode = _resolve_cache_mode(cache, device, data_yaml)

    print(f"\n🚀 Pornesc antrenamentul...")
    print("-" * 60)

    # Antrenare
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(Path('runs') / 'detect'),
        name=project_name,
        exist_ok=True,
        patience=30,  # Early stopping
        save=True,
        plots=True,
        verbose=True,
        workers=workers,
        amp=True,  # Mixed precision (mai rapid pe CUDA)
        cache=cache_mode,
        resume=resume,
    )
    
    print("\n" + "=" * 60)
    print("✅ ANTRENAMENT FINALIZAT!")
    print("=" * 60)
    
    # Afișează rezultatele (folosește direct save_dir din Ultralytics)
    results_dir = Path(str(getattr(results, 'save_dir', '')))

    if str(results_dir) and results_dir.exists():
        print(f"\n📁 Rezultate salvate în: {results_dir}")
        print(f"   📊 Weights: {results_dir / 'weights' / 'best.pt'}")
        print(f"   📈 Grafice: {results_dir / 'results.png'}")
    else:
        print("\n📁 Rezultate salvate (Ultralytics):", getattr(results, 'save_dir', '(necunoscut)'))

    return results


def main():
    """Funcția principală cu argumente CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🌿 Antrenament YOLOv8 pentru seră hidroponică (busuioc/mentă sau dataset custom)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple:
  python train.py                           # Antrenament default (100 epoci, YOLOv8n)
  python train.py --epochs 200              # 200 epoci
  python train.py --model s --epochs 150    # YOLOv8 Small, 150 epoci
  python train.py --data custom/data.yaml   # Dataset custom
        """
    )
    
    parser.add_argument('--data', type=str, default=None,
                        help='Calea către data.yaml')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Numărul de epoci (default: 100)')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Dimensiunea modelului: n(ano), s(mall), m(edium), l(arge), x(large)')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (auto dacă nespecificat)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Dimensiunea imaginilor (default: 640)')
    parser.add_argument('--name', type=str, default='herbs_yolo',
                        help='Numele run-ului (folder sub runs\\detect\\)')
    parser.add_argument('--resume', action='store_true',
                        help='Continuă antrenamentul anterior')
    parser.add_argument('--cache', type=str, default='auto',
                        choices=['auto', 'ram', 'disk', 'false'],
                        help='Cache imagini: auto/ram/disk/false (default: auto)')
    
    args = parser.parse_args()
    
    train(
        data_yaml=args.data,
        epochs=args.epochs,
        model_size=args.model,
        batch_size=args.batch,
        imgsz=args.imgsz,
        project_name=args.name,
        resume=args.resume,
        cache=args.cache,
    )


if __name__ == "__main__":
    main()
