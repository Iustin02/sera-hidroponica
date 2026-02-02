#!/usr/bin/env python3
"""
=============================================================================
Proiect de Licență: Seră Hidroponică Smart
Script de Antrenament YOLOv8 Nano pentru Detectarea Sănătății Plantelor
Autor: [Numele tău]
Data: Februarie 2026
=============================================================================

Descriere:
    Acest script antrenează un model YOLOv8 Nano pentru detectarea automată
    a problemelor de sănătate la roșiile Micro Tom într-o seră hidroponică.
    
    Modelul este optimizat pentru rulare pe Apple Silicon (M1 Pro) folosind
    Metal Performance Shaders (MPS) pentru accelerare GPU.
"""

# =============================================================================
# IMPORTURI BIBLIOTECI
# =============================================================================
# Importăm biblioteca ultralytics care conține implementarea YOLOv8
from ultralytics import YOLO

# Importăm biblioteca os pentru operații cu sistemul de fișiere
import os

# Importăm torch pentru verificarea disponibilității GPU-ului
import torch

# Importăm datetime pentru timestamp-ul modelului salvat
from datetime import datetime


# =============================================================================
# CONFIGURARE PARAMETRI DE ANTRENAMENT
# =============================================================================

# Calea către fișierul de configurare a setului de date
# Acest fișier YAML conține căile către imagini și clasele de detectat
DATA_CONFIG = "data.yaml"

# Numărul de epoci de antrenament
# O epocă = o parcurgere completă a întregului set de date
EPOCHS = 50

# Dimensiunea imaginilor pentru antrenament (în pixeli)
# YOLOv8 redimensionează automat imaginile la această rezoluție
IMAGE_SIZE = 640

# Device-ul pentru antrenament
# 'mps' = Metal Performance Shaders (GPU Apple Silicon M1/M2/M3)
# Alternativ: 'cpu' pentru procesor, 'cuda' pentru GPU NVIDIA
DEVICE = "mps"

# Modelul pre-antrenat YOLOv8 Nano
# Variante disponibile: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
# 'n' = Nano (cel mai mic și rapid, ideal pentru edge devices)
MODEL_PRETRAINED = "yolov8n.pt"

# Directorul pentru salvarea rezultatelor antrenamentului
OUTPUT_DIR = "runs/train"


# =============================================================================
# FUNCȚIE PENTRU VERIFICAREA SUPORTULUI MPS
# =============================================================================
def verifica_mps():
    """
    Verifică dacă GPU-ul Apple Silicon (MPS) este disponibil.
    
    Returnează:
        bool: True dacă MPS este disponibil, False altfel
    """
    # Verificăm dacă PyTorch a fost compilat cu suport MPS
    if not torch.backends.mps.is_built():
        print("⚠️  PyTorch nu a fost compilat cu suport MPS!")
        return False
    
    # Verificăm dacă MPS este disponibil pe acest sistem
    if not torch.backends.mps.is_available():
        print("⚠️  MPS nu este disponibil pe acest sistem!")
        return False
    
    print("✅ GPU Apple Silicon (MPS) detectat și disponibil!")
    return True


# =============================================================================
# FUNCȚIE PRINCIPALĂ DE ANTRENAMENT
# =============================================================================
def antreneaza_model():
    """
    Funcția principală care realizează antrenamentul modelului YOLOv8.
    
    Pași:
        1. Verifică disponibilitatea GPU-ului M1 Pro
        2. Încarcă modelul pre-antrenat YOLOv8 Nano
        3. Configurează și pornește antrenamentul
        4. Salvează modelul antrenat
    
    Returnează:
        str: Calea către modelul antrenat salvat
    """
    
    print("=" * 70)
    print("🌱 SERĂ HIDROPONICĂ SMART - Antrenament Model YOLOv8")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # PASUL 1: Verificare disponibilitate GPU M1 Pro (MPS)
    # -------------------------------------------------------------------------
    print("\n📍 Pasul 1: Verificare GPU Apple Silicon...")
    
    # Selectăm device-ul de antrenament
    if DEVICE == "mps" and verifica_mps():
        device_folosit = "mps"
        print(f"   └── Se va folosi: GPU M1 Pro (Metal Performance Shaders)")
    else:
        device_folosit = "cpu"
        print(f"   └── Se va folosi: CPU (fallback)")
    
    # -------------------------------------------------------------------------
    # PASUL 2: Încărcare model pre-antrenat YOLOv8 Nano
    # -------------------------------------------------------------------------
    print(f"\n📍 Pasul 2: Încărcare model pre-antrenat...")
    print(f"   └── Model: {MODEL_PRETRAINED}")
    
    # Încărcăm modelul pre-antrenat
    # Dacă fișierul nu există local, se descarcă automat de pe serverele Ultralytics
    model = YOLO(MODEL_PRETRAINED)
    
    print("   └── ✅ Model încărcat cu succes!")
    
    # -------------------------------------------------------------------------
    # PASUL 3: Verificare fișier de configurare dataset
    # -------------------------------------------------------------------------
    print(f"\n📍 Pasul 3: Verificare configurare dataset...")
    
    if not os.path.exists(DATA_CONFIG):
        print(f"   └── ❌ EROARE: Fișierul {DATA_CONFIG} nu a fost găsit!")
        print(f"   └── Te rog să creezi fișierul data.yaml cu configurarea datasetului.")
        return None
    
    print(f"   └── ✅ Fișier configurare: {DATA_CONFIG}")
    
    # -------------------------------------------------------------------------
    # PASUL 4: Pornire antrenament
    # -------------------------------------------------------------------------
    print(f"\n📍 Pasul 4: Pornire antrenament...")
    print(f"   ├── Epoci: {EPOCHS}")
    print(f"   ├── Dimensiune imagine: {IMAGE_SIZE}x{IMAGE_SIZE} pixeli")
    print(f"   ├── Device: {device_folosit}")
    print(f"   └── Dataset: {DATA_CONFIG}")
    print("\n" + "-" * 70)
    print("🚀 Antrenamentul începe... (poate dura câteva ore)")
    print("-" * 70 + "\n")
    
    # Pornim antrenamentul cu parametrii configurați
    # Metoda train() returnează un obiect Results cu metrici și statistici
    rezultate = model.train(
        data=DATA_CONFIG,           # Calea către fișierul YAML cu configurarea datasetului
        epochs=EPOCHS,              # Numărul total de epoci de antrenament
        imgsz=IMAGE_SIZE,           # Dimensiunea imaginilor (pătrată)
        device=device_folosit,      # Device-ul pentru calcule (mps/cpu/cuda)
        project=OUTPUT_DIR,         # Directorul părinte pentru salvare
        name="sera_hidroponica",    # Numele experimentului
        exist_ok=True,              # Permite suprascrierea dacă există
        verbose=True,               # Afișează informații detaliate
        patience=10,                # Early stopping: oprește dacă nu se îmbunătățește 10 epoci
        save=True,                  # Salvează checkpoint-uri în timpul antrenamentului
        plots=True                  # Generează grafice cu metrici
    )
    
    # -------------------------------------------------------------------------
    # PASUL 5: Salvare model antrenat
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("📍 Pasul 5: Salvare model antrenat...")
    
    # Calea către modelul cel mai bun (best.pt)
    model_best_path = os.path.join(OUTPUT_DIR, "sera_hidroponica", "weights", "best.pt")
    
    # Calea către ultimul model (last.pt)
    model_last_path = os.path.join(OUTPUT_DIR, "sera_hidroponica", "weights", "last.pt")
    
    # Generăm un nume personalizat cu timestamp pentru backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_backup_name = f"model_sera_hidroponica_{timestamp}.pt"
    model_backup_path = os.path.join(".", model_backup_name)
    
    # Copiem modelul best.pt cu nume personalizat în directorul curent
    if os.path.exists(model_best_path):
        import shutil
        shutil.copy(model_best_path, model_backup_path)
        print(f"   ├── Model 'best' salvat: {model_best_path}")
        print(f"   ├── Backup creat: {model_backup_path}")
    
    print(f"   └── ✅ Modele salvate cu succes!")
    
    # -------------------------------------------------------------------------
    # PASUL 6: Afișare rezumat și metrici
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 REZUMAT ANTRENAMENT")
    print("=" * 70)
    print(f"   ├── Directorul rezultatelor: {OUTPUT_DIR}/sera_hidroponica/")
    print(f"   ├── Model best.pt: {model_best_path}")
    print(f"   ├── Model last.pt: {model_last_path}")
    print(f"   └── Grafice și metrici: {OUTPUT_DIR}/sera_hidroponica/")
    print("\n✅ Antrenamentul s-a finalizat cu succes!")
    print("=" * 70)
    
    return model_best_path


# =============================================================================
# PUNCT DE INTRARE PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    """
    Punct de intrare când scriptul este rulat direct.
    
    Utilizare:
        python train_hydro.py
    
    Cerințe:
        - Mediul virtual activat (source venv/bin/activate)
        - Fișierul data.yaml configurat corect
        - Imaginile și etichetele în directoarele specificate
    """
    
    # Afișăm informații despre versiuni
    print("\n🔧 Verificare versiuni biblioteci:")
    print(f"   ├── PyTorch: {torch.__version__}")
    
    try:
        import ultralytics
        print(f"   └── Ultralytics: {ultralytics.__version__}")
    except:
        print("   └── Ultralytics: versiune necunoscută")
    
    # Pornim antrenamentul
    cale_model = antreneaza_model()
    
    if cale_model:
        print(f"\n💡 Pentru a folosi modelul antrenat:")
        print(f"   >>> from ultralytics import YOLO")
        print(f"   >>> model = YOLO('{cale_model}')")
        print(f"   >>> results = model.predict('imagine_test.jpg')")
