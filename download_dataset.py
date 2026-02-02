#!/usr/bin/env python3
"""
Script pentru descărcarea dataset-ului de boli la frunze de roșii
de pe Roboflow Universe - format YOLOv8 ready
"""

from roboflow import Roboflow

# =============================================================================
# OPȚIUNEA 1: Dataset PUBLIC de pe Roboflow Universe (fără API key)
# Folosim export direct prin URL
# =============================================================================

import os
import urllib.request
import zipfile

def descarcare_dataset_public():
    """
    Descarcă un dataset public de boli la frunze de roșii
    în format YOLOv8 gata de antrenament.
    """
    
    print("=" * 60)
    print("🌿 Descărcare Dataset - Tomato Leaf Disease")
    print("=" * 60)
    
    # Directorul destinație
    dataset_dir = "./dataset_tomato_disease"
    
    print(f"\n📁 Dataset va fi salvat în: {dataset_dir}")
    print("\n⏳ Pentru a descărca dataset-ul, urmează acești pași:\n")
    
    print("=" * 60)
    print("📋 INSTRUCȚIUNI DESCĂRCARE MANUALĂ (Recomandat)")
    print("=" * 60)
    
    print("""
    1. Accesează unul din aceste link-uri:
    
       🔹 Dataset mare (8.6k imagini, 10 clase):
       https://universe.roboflow.com/universitas-atma-jaya/tomato-leaf-disease-rxcft
       
       🔹 Dataset mediu (4k imagini, 10 clase):
       https://universe.roboflow.com/bryan-b56jm/tomato-leaf-disease-ssoha
       
       🔹 Dataset mic pentru test (739 imagini, 8 clase):
       https://universe.roboflow.com/student-nrpvc/tomato-leaf-disease-c3xkq
    
    2. Click pe "Download Dataset"
    
    3. Selectează formatul: YOLOv8
    
    4. Descarcă și dezarhivează în ~/sera-hidroponica/dataset/
    
    5. Structura trebuie să fie:
       dataset/
       ├── images/
       │   ├── train/
       │   └── val/
       ├── labels/
       │   ├── train/
       │   └── val/
       └── data.yaml
    """)
    
    print("=" * 60)
    print("📋 ALTERNATIVĂ: Descărcare cu API Key Roboflow")
    print("=" * 60)
    
    print("""
    1. Creează cont gratuit pe https://roboflow.com
    
    2. Obține API Key-ul din Settings > API Key
    
    3. Rulează codul următor în Python:
    
    >>> from roboflow import Roboflow
    >>> rf = Roboflow(api_key="YOUR_API_KEY")
    >>> project = rf.workspace("universitas-atma-jaya").project("tomato-leaf-disease-rxcft")
    >>> version = project.version(2)
    >>> dataset = version.download("yolov8")
    """)
    
    print("\n" + "=" * 60)
    print("✅ După descărcare, actualizează data.yaml cu calea corectă!")
    print("=" * 60)


def descarcare_cu_api_key(api_key):
    """
    Descarcă dataset-ul folosind API key-ul Roboflow.
    
    Args:
        api_key: API key-ul tău de pe Roboflow
    """
    
    print("🔄 Conectare la Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    print("📦 Accesare proiect Tomato Leaf Disease...")
    # Dataset cu 8.4k imagini și 11 clase
    project = rf.workspace("universitas-atma-jaya").project("tomato-leaf-disease-rxcft")
    
    print("⬇️  Descărcare versiune în format YOLOv8...")
    version = project.version(2)
    dataset = version.download("yolov8", location="./dataset_roboflow")
    
    print(f"\n✅ Dataset descărcat în: {dataset.location}")
    print("\n📝 Clasele din dataset:")
    
    return dataset.location


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Dacă ai API key, îl poți pasa ca argument
        api_key = sys.argv[1]
        descarcare_cu_api_key(api_key)
    else:
        # Afișează instrucțiunile de descărcare manuală
        descarcare_dataset_public()
