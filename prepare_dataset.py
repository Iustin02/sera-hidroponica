#!/usr/bin/env python3
"""
Script pentru descărcarea și pregătirea dataset-ului PlantVillage
cu imagini de frunze de roșii pentru antrenament YOLOv8

Acest script:
1. Descarcă un subset de imagini cu frunze de roșii
2. Le organizează în structura YOLOv8 (images/train, images/val, labels/train, labels/val)
3. Creează automat etichetele pentru clasificare
"""

import os
import shutil
import random
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Directorul de bază
BASE_DIR = Path("./tomato_leaf_dataset")
DATASET_DIR = Path("./dataset")

# Clasele pentru detectarea bolilor la roșii
CLASSES = {
    0: "Tomato_healthy",
    1: "Tomato_Bacterial_spot", 
    2: "Tomato_Early_blight",
    3: "Tomato_Late_blight",
    4: "Tomato_Leaf_Mold",
    5: "Tomato_Septoria_leaf_spot",
    6: "Tomato_Spider_mites",
    7: "Tomato_Target_Spot",
    8: "Tomato_Yellow_Leaf_Curl_Virus",
    9: "Tomato_mosaic_virus"
}

def download_sample_images():
    """
    Descarcă imagini sample din surse publice pentru demonstrație.
    Pentru antrenament real, folosește dataset-ul complet PlantVillage.
    """
    
    print("=" * 60)
    print("🍅 Pregătire Dataset Demonstrativ - Tomato Leaf Disease")
    print("=" * 60)
    
    # Creăm structura de directoare
    for split in ['train', 'val']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("\n✅ Structură directoare creată!")
    
    # Imagini de test de pe Unsplash/Pexels (frunze verzi generice pentru demonstrație)
    # Pentru proiectul real, descarcă PlantVillage dataset
    sample_urls = [
        # Aceste URL-uri sunt placeholder - pentru demo
        ("https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=640", "leaf_sample_1.jpg"),
        ("https://images.unsplash.com/photo-1520412099551-62b6bafeb5bb?w=640", "leaf_sample_2.jpg"),
        ("https://images.unsplash.com/photo-1598512752271-33f913a5af13?w=640", "leaf_sample_3.jpg"),
    ]
    
    print("\n📥 Pentru dataset-ul COMPLET, urmează acești pași:")
    print("-" * 60)
    print("""
    OPȚIUNEA 1 - Kaggle (Recomandat):
    ─────────────────────────────────
    1. Accesează: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
    2. Click "Download" (necesită cont Kaggle gratuit)
    3. Dezarhivează fișierul
    4. Copiază folderul 'tomato' în ~/sera-hidroponica/raw_data/
    5. Rulează: python prepare_yolo_dataset.py
    
    OPȚIUNEA 2 - Roboflow (Deja format YOLO):
    ─────────────────────────────────────────
    1. Accesează: https://universe.roboflow.com/universitas-atma-jaya/tomato-leaf-disease-rxcft
    2. Click "Download Dataset" → Format: YOLOv8
    3. Dezarhivează direct în ~/sera-hidroponica/dataset/
    
    OPȚIUNEA 3 - GitHub (Direct):
    ─────────────────────────────
    Rulează în terminal:
    git clone https://github.com/spMohanty/PlantVillage-Dataset.git raw_data
    """)
    
    return True


def create_sample_dataset():
    """
    Creează un dataset mic de demonstrație cu imagini generate.
    """
    import numpy as np
    from PIL import Image
    
    print("\n🎨 Generez imagini sintetice pentru test rapid...")
    
    # Generăm imagini simple pentru a testa pipeline-ul
    for split in ['train', 'val']:
        num_images = 20 if split == 'train' else 5
        
        for i in range(num_images):
            # Generăm o imagine random (simulează o frunză)
            # Verde pentru sănătos, maro pentru bolnav
            class_id = random.randint(0, 3)
            
            if class_id == 0:  # Sănătos - verde
                color = (34, 139, 34)  # Forest green
            elif class_id == 1:  # Bacterial spot - pete maro
                color = (139, 90, 43)
            elif class_id == 2:  # Early blight - galben-maro
                color = (218, 165, 32)
            else:  # Late blight - maro închis
                color = (101, 67, 33)
            
            # Creăm imaginea
            img = Image.new('RGB', (640, 640), color)
            
            # Adăugăm variație
            pixels = np.array(img)
            noise = np.random.randint(-20, 20, pixels.shape, dtype=np.int16)
            pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(pixels)
            
            # Salvăm imaginea
            img_name = f"tomato_{split}_{i:04d}.jpg"
            img_path = DATASET_DIR / 'images' / split / img_name
            img.save(img_path, quality=95)
            
            # Creăm eticheta YOLO (bounding box pe toată imaginea pentru clasificare)
            # Format: class_id x_center y_center width height (normalizate 0-1)
            label_name = f"tomato_{split}_{i:04d}.txt"
            label_path = DATASET_DIR / 'labels' / split / label_name
            
            # Bounding box aleator în imagine
            x_center = random.uniform(0.3, 0.7)
            y_center = random.uniform(0.3, 0.7)
            width = random.uniform(0.3, 0.5)
            height = random.uniform(0.3, 0.5)
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"   ├── Train: 20 imagini generate")
    print(f"   └── Val: 5 imagini generate")
    
    return True


def create_data_yaml():
    """
    Creează fișierul data.yaml pentru YOLOv8.
    """
    
    yaml_content = """# Dataset YOLOv8 - Tomato Leaf Disease Detection
# Generat automat pentru proiectul Seră Hidroponică Smart

path: ./dataset
train: images/train
val: images/val

# Numărul de clase
nc: 4

# Denumirea claselor (subset pentru demo)
names:
  0: healthy
  1: bacterial_spot
  2: early_blight
  3: late_blight

# Note:
# - Pentru antrenament real, descarcă dataset-ul complet PlantVillage
# - Acest dataset demo este doar pentru testarea pipeline-ului
"""
    
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("\n✅ Fișier data.yaml creat!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌱 SERĂ HIDROPONICĂ - Pregătire Dataset")
    print("=" * 60)
    
    # Afișăm instrucțiunile pentru dataset-ul real
    download_sample_images()
    
    # Întrebăm utilizatorul
    print("\n" + "=" * 60)
    response = input("🤔 Vrei să generez un dataset DEMO pentru a testa pipeline-ul? (y/n): ")
     
    if response.lower() in ['y', 'yes', 'da']:
        create_sample_dataset()
        create_data_yaml()
        
        print("\n" + "=" * 60)
        print("✅ Dataset demo creat cu succes!")
        print("=" * 60)
        print("\n📝 Poți rula acum antrenamentul cu:")
        print("   python train_hydro.py")
        print("\n⚠️  ATENȚIE: Acest dataset este doar pentru TEST!")
        print("   Pentru rezultate reale, descarcă PlantVillage dataset.")
    else:
        print("\n👉 Urmează instrucțiunile de mai sus pentru dataset-ul real.")
