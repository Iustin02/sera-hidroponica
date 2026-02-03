#!/usr/bin/env python3
"""
🍅 DESCĂRCARE AUTOMATĂ DATASET PLANTVILLAGE TOMATO
Script pentru descărcarea și pregătirea dataset-ului complet pentru YOLOv8

Autor: Iustin
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm

# Setări pentru D:
os.environ['TEMP'] = 'D:\\temp'
os.environ['TMP'] = 'D:\\temp'

# Configurație
OUTPUT_DIR = Path("d:/sera-hidroponica/plantvillage_full")
KAGGLE_DATASET = "emmarex/plantdisease"


def download_from_kaggle():
    """
    Descarcă dataset-ul PlantVillage de pe Kaggle.
    Necesită configurare API key Kaggle.
    """
    print("\n📥 Descărcare de pe Kaggle...")
    print("=" * 60)
    
    # Verifică dacă avem API key configurat
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        print("""
❌ API Key Kaggle nu este configurat!

Pentru a configura:
1. Accesează https://www.kaggle.com/settings
2. Click pe "Create New Token" în secțiunea API
3. Fișierul kaggle.json se va descărca automat
4. Mută fișierul în: {kaggle_json.parent}
5. Rulează din nou acest script

SAU descarcă manual:
1. https://www.kaggle.com/datasets/emmarex/plantdisease
2. Dezarhivează în: d:/sera-hidroponica/plantvillage_raw/
3. Rulează din nou acest script
        """)
        return None
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        download_dir = Path("d:/sera-hidroponica/plantvillage_raw")
        download_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Descărcare {KAGGLE_DATASET}...")
        api.dataset_download_files(KAGGLE_DATASET, path=str(download_dir), unzip=True)
        
        print("✅ Descărcare completă!")
        return download_dir
        
    except Exception as e:
        print(f"❌ Eroare la descărcare: {e}")
        return None


def find_tomato_classes(raw_dir):
    """
    Găsește toate clasele de frunze de roșii din dataset.
    """
    tomato_classes = {}
    
    # Căutăm în toate directoarele
    for item in raw_dir.rglob("*"):
        if item.is_dir():
            name = item.name.lower()
            if 'tomato' in name:
                # Extragem numele bolii
                parts = item.name.replace('___', '_').replace('__', '_').split('_')
                
                # Identificăm clasa
                if 'healthy' in name:
                    class_name = 'healthy'
                elif 'bacterial' in name and 'spot' in name:
                    class_name = 'bacterial_spot'
                elif 'early' in name and 'blight' in name:
                    class_name = 'early_blight'
                elif 'late' in name and 'blight' in name:
                    class_name = 'late_blight'
                elif 'leaf' in name and 'mold' in name:
                    class_name = 'leaf_mold'
                elif 'septoria' in name:
                    class_name = 'septoria_leaf_spot'
                elif 'spider' in name or 'mite' in name:
                    class_name = 'spider_mites'
                elif 'target' in name and 'spot' in name:
                    class_name = 'target_spot'
                elif 'yellow' in name and 'curl' in name:
                    class_name = 'yellow_leaf_curl'
                elif 'mosaic' in name:
                    class_name = 'mosaic_virus'
                else:
                    continue
                
                # Numărăm imaginile
                images = list(item.glob("*.jpg")) + list(item.glob("*.JPG")) + list(item.glob("*.png"))
                if images:
                    if class_name not in tomato_classes:
                        tomato_classes[class_name] = []
                    tomato_classes[class_name].extend(images)
    
    return tomato_classes


def convert_to_yolo(raw_dir, output_dir, train_split=0.8):
    """
    Convertește dataset-ul PlantVillage în format YOLO.
    """
    print("\n🔄 Conversie în format YOLO...")
    print("=" * 60)
    
    # Găsim clasele
    tomato_classes = find_tomato_classes(raw_dir)
    
    if not tomato_classes:
        print("❌ Nu am găsit clase de frunze de roșii!")
        print(f"   Verifică structura în: {raw_dir}")
        return None
    
    # Afișăm statistici
    print("\n📊 Clase găsite:")
    total_images = 0
    for class_name, images in sorted(tomato_classes.items()):
        print(f"   {class_name}: {len(images)} imagini")
        total_images += len(images)
    print(f"\n   TOTAL: {total_images} imagini")
    
    # Creăm directoarele
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Mapare clase
    class_names = sorted(tomato_classes.keys())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\n📋 Mapare clase -> ID:")
    for name, idx in class_to_id.items():
        print(f"   {idx}: {name}")
    
    # Procesăm fiecare clasă
    print(f"\n🔄 Procesare imagini...")
    
    train_count = 0
    val_count = 0
    
    for class_name, images in tomato_classes.items():
        class_id = class_to_id[class_name]
        
        # Shuffle
        random.shuffle(images)
        
        # Split
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Procesăm train
        for img_path in tqdm(train_images, desc=f"Train {class_name}"):
            try:
                new_name = f"{class_name}_{train_count:05d}"
                
                # Copiem și convertim imaginea
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize((640, 640), Image.LANCZOS)  # Resize la 640x640
                img.save(output_dir / "images" / "train" / f"{new_name}.jpg", quality=95)
                
                # Label YOLO (bbox pentru întreaga imagine)
                with open(output_dir / "labels" / "train" / f"{new_name}.txt", 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.95 0.95\n")
                
                train_count += 1
            except Exception as e:
                print(f"\n⚠️  Eroare la {img_path}: {e}")
        
        # Procesăm val
        for img_path in tqdm(val_images, desc=f"Val {class_name}"):
            try:
                new_name = f"{class_name}_{val_count:05d}"
                
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize((640, 640), Image.LANCZOS)
                img.save(output_dir / "images" / "val" / f"{new_name}.jpg", quality=95)
                
                with open(output_dir / "labels" / "val" / f"{new_name}.txt", 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.95 0.95\n")
                
                val_count += 1
            except Exception as e:
                print(f"\n⚠️  Eroare la {img_path}: {e}")
    
    # Creăm data.yaml
    data_yaml = output_dir / "data.yaml"
    yaml_content = f"""# PlantVillage Tomato Leaf Disease Dataset
# Generat automat pentru antrenament YOLOv8
# Total: {train_count + val_count} imagini

path: {output_dir.as_posix()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}

# Statistici:
# Train: {train_count} imagini
# Val: {val_count} imagini
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Conversie completă!")
    print(f"   Output: {output_dir}")
    print(f"   Train: {train_count} imagini")
    print(f"   Val: {val_count} imagini")
    print(f"   data.yaml: {data_yaml}")
    
    return output_dir


def download_alternative():
    """
    Descarcă un dataset alternativ de pe internet.
    """
    print("\n📥 Descărcare dataset alternativ...")
    print("=" * 60)
    
    # URL-uri pentru dataset-uri publice
    urls = [
        # Dataset PlantVillage subset pe GitHub
        "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip",
    ]
    
    download_dir = Path("d:/sera-hidroponica/plantvillage_raw")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = download_dir / "plantvillage.zip"
    
    for url in urls:
        try:
            print(f"   Încercare: {url[:60]}...")
            
            # Descărcare cu progress
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
                print(f"\r   Descărcare: {percent}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, zip_path, reporthook)
            print("\n   ✅ Descărcare completă!")
            
            # Dezarhivare
            print("   📦 Dezarhivare...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            
            # Ștergem zip-ul
            zip_path.unlink()
            
            return download_dir
            
        except Exception as e:
            print(f"\n   ❌ Eroare: {e}")
            continue
    
    print("\n❌ Nu s-a putut descărca dataset-ul automat.")
    print("   Descarcă manual de pe Kaggle și pune în plantvillage_raw/")
    return None


def main():
    """Funcția principală."""
    
    print("\n" + "=" * 70)
    print("🌿 DESCĂRCARE DATASET PLANTVILLAGE - TOMATO LEAF DISEASES")
    print("   Pentru antrenament YOLOv8 pe RTX 3060")
    print("=" * 70)
    
    # Verificăm dacă dataset-ul există deja
    if OUTPUT_DIR.exists():
        train_dir = OUTPUT_DIR / "images" / "train"
        if train_dir.exists():
            num_images = len(list(train_dir.glob("*.jpg")))
            if num_images > 1000:
                print(f"\n✅ Dataset deja pregătit: {num_images} imagini train")
                print(f"   Locație: {OUTPUT_DIR}")
                print(f"   data.yaml: {OUTPUT_DIR / 'data.yaml'}")
                return OUTPUT_DIR
    
    # Verificăm dacă există date raw
    raw_dir = Path("d:/sera-hidroponica/plantvillage_raw")
    
    if raw_dir.exists():
        # Căutăm subdirectoare cu imagini
        has_data = False
        for item in raw_dir.rglob("*"):
            if item.is_dir() and 'tomato' in item.name.lower():
                images = list(item.glob("*.jpg")) + list(item.glob("*.JPG"))
                if images:
                    has_data = True
                    break
        
        if has_data:
            print(f"\n📁 Date găsite în: {raw_dir}")
            return convert_to_yolo(raw_dir, OUTPUT_DIR)
    
    # Încercăm descărcarea
    print("\n📥 Dataset-ul nu există. Opțiuni:")
    print("   1. Descărcare automată de pe GitHub (subset ~4000 imagini)")
    print("   2. Descărcare de pe Kaggle (complet ~18000 imagini)")
    print("   3. Ieșire (descărcare manuală)")
    
    choice = input("\nAlege opțiunea (1/2/3): ").strip()
    
    if choice == '1':
        raw_dir = download_alternative()
        if raw_dir:
            return convert_to_yolo(raw_dir, OUTPUT_DIR)
    
    elif choice == '2':
        raw_dir = download_from_kaggle()
        if raw_dir:
            return convert_to_yolo(raw_dir, OUTPUT_DIR)
    
    else:
        print("""
📋 INSTRUCȚIUNI DESCĂRCARE MANUALĂ:

1. Accesează: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Click "Download" (necesită cont gratuit)
3. Dezarhivează în: d:/sera-hidroponica/plantvillage_raw/
4. Structura trebuie să fie:
   plantvillage_raw/
   └── PlantVillage/
       ├── Tomato___healthy/
       ├── Tomato___Bacterial_spot/
       ├── Tomato___Early_blight/
       └── ...
5. Rulează din nou: python download_plantvillage.py
        """)
    
    return None


if __name__ == "__main__":
    result = main()
    
    if result:
        print("\n" + "=" * 70)
        print("✅ DATASET PREGĂTIT!")
        print("=" * 70)
        print(f"\nPentru antrenament cu 1000 epoci, rulează:")
        print(f"   python train_plantvillage_1000.py")
        print(f"\nSau direct cu train.py:")
        print(f"   python train.py --data {result / 'data.yaml'} --epochs 1000")
