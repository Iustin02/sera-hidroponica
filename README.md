# 🍅 Sera Hidroponică Smart - Detecție Boli Frunze cu YOLOv8

Proiect de licență pentru detecția automată a bolilor la roșii Micro Tom într-o seră hidroponică smart.

## 🚀 Quick Start

### 1. Clonează repository-ul
```bash
git clone https://github.com/username/sera-hidroponica.git
cd sera-hidroponica
```

### 2. Creează environment-ul virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

### 3. Instalează dependențele

**Pentru NVIDIA GPU (CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Pentru Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision
pip install -r requirements.txt
```

**Pentru CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Rulează antrenamentul
```bash
python train.py
```

---

## 📊 Rezultate

| Metrică | Valoare |
|---------|---------|
| mAP@50 | 81.5% |
| mAP@50-95 | 75.4% |
| Precision | 73.3% |
| Recall | 85.6% |

### Performanță per clasă:
- ✅ healthy: 99.5%
- ✅ late_blight: 99.2%
- ✅ early_blight: 95.6%
- ✅ leaf_mold: 89.7%
- ⚠️ bacterial_spot: 60.7%
- ⚠️ septoria_leaf_spot: 44.1%

---

## 🖥️ Compatibilitate Hardware

| Hardware | Suportat | Batch Size Recomandat |
|----------|----------|----------------------|
| NVIDIA RTX 3060 12GB | ✅ | 32 |
| NVIDIA RTX 3070/3080 | ✅ | 32-64 |
| Apple M1/M2/M3 | ✅ | 8-16 |
| Intel/AMD CPU | ✅ (lent) | 4-8 |

---

## 📁 Structura Proiectului

```
sera-hidroponica/
├── train.py              # Script principal antrenament (hibrid CUDA/MPS/CPU)
├── requirements.txt      # Dependențe Python
├── data.yaml            # Configurație dataset
├── tomato_real/         # Dataset
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── runs/                # Rezultate antrenament
│   └── detect/
│       └── tomato_disease/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── results.csv
└── README.md
```

---

## 🎯 Utilizare

### Antrenament basic:
```bash
python train.py
```

### Antrenament cu opțiuni:
```bash
# 200 epoci cu model Small
python train.py --epochs 200 --model s

# Dataset custom
python train.py --data path/to/data.yaml

# Batch size specific
python train.py --batch 64 --epochs 150
```

### Inferență pe imagini noi:
```python
from ultralytics import YOLO

model = YOLO('runs/detect/tomato_disease/weights/best.pt')
results = model('path/to/image.jpg')
results[0].show()
```

---

## 🔧 Troubleshooting

### CUDA out of memory
Reduceți batch size:
```bash
python train.py --batch 8
```

### MPS (Apple) prea lent
Este normal, MPS e experimental. Pentru antrenament serios, folosiți CUDA.

### Nu găsește data.yaml
Verificați structura dataset-ului și calea:
```bash
python train.py --data tomato_real/data.yaml
```

---

## 📚 Resurse

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [Roboflow Universe](https://universe.roboflow.com/)

---

## 👤 Autor

**Iustin Iulian Dodita**  
Proiect licență - Seră Hidroponică Smart

---

## 📄 Licență

MIT License - Folosiți liber pentru proiecte educaționale și personale.
