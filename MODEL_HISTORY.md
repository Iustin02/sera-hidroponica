# Model History — Sera Hidroponica / Edge AI NFT

Jurnal de evoluție a modelului: de unde am pornit, ce am schimbat, de ce, și ce rezultate am obținut.

---

## Punct de pornire — starea inițială a proiectului

**Data:** 2026-06-11

Proiectul exista deja cu mai multe fișiere și scripturi de antrenare, dar fără o direcție clară și fără un model funcțional.

### Dataset-uri disponibile la start

| Dataset | Tip | Clase | Imagini |
|---|---|---|---|
| `basil_dataset/` | Detection (YOLO bbox) | 5 clase | ~800 train |
| `basil_classification_dataset/` | Classification (foldere) | 3 clase | 941 train |

**Clase în `basil_classification_dataset` (3 clase — Roboflow):**
- `healthy`, `fusarium`, `powdery`

### Model existent la start

Script: `train_basil_complete.py` → `yolov8n.pt` pe `basil_dataset` (5 clase), **3 epoci** (nefuncțional)

| Metrică | Valoare |
|---|---|
| mAP50 | 0.076 |
| Precision | 0.666 |
| Recall | 0.199 |

---

## Decizie arhitecturală — schema finală de 3 clase

**Schema aleasă** (`BASIL_DETECTION_CLASS_SCHEMA.md`):

| ID | Clasă | Denumire simplă |
|---|---|---|
| 0 | `healthy` | sănătoasă |
| 1 | `Fusarium oxysporum` | fusarioză |
| 2 | `Oidium belbahrii` | făinare |

**Mapping control:**
- `healthy` → nicio acțiune
- `Fusarium oxysporum` → alertă + izolare + verificare pH/EC
- `Oidium belbahrii` → creștere ventilație, scădere umiditate sub 60%

---

## Ramura Detection (abandonată temporar)

### Trial D1 — Detection v1 (full-image boxes)

**Run:** `runs/detect/runs/detect/basil_3class/`

| Parametru | Valoare |
|---|---|
| Model | `yolov8n.pt` |
| Box size | 1.0×1.0 (full-image) |
| Mosaic | activat |
| Epoci | 25/100 (early stop) |

| Metrică | Valoare |
|---|---|
| mAP50 | 0.654 |
| Precision | 0.477 ❌ |
| Recall | 0.861 ✅ |

**Problemă:** Precision scăzut structural — mosaic + full-image boxes cauzează false positives.

### Trial D2 — Detection v2 (fix precision)

**Run:** `runs/detect/runs/detect/basil_3class_v2/`

| Modificare | Valoare |
|---|---|
| Box size | 0.85×0.85 |
| Mosaic | 0.0 (dezactivat) |
| Box loss | 3.0 (↓ de la 7.5) |
| Cls loss | 1.5 (↑ de la 0.5) |

| Metrică | Valoare |
|---|---|
| mAP50 | **0.443** ❌ (mai slab decât D1) |
| Precision | 0.368 |

**Concluzie:** Dezactivarea mosaic a eliminat augmentarea principală → model mai slab. Detection cu full-image boxes are un plafon structural și nu poate atinge precision ridicat fără bbox-uri reale adnotate.

**Decizie:** trecem la **clasificare** ca abordare principală. Detection cu bbox-uri reale rămâne ca plan pe termen lung (Pasul 2 — adnotare manuală în Roboflow).

---

## Ramura Classification (abordare principală)

### Trial C1 — Classification nano, imgsz=224

**Run:** `runs/classify/runs/classify/basil_health_v1/`

| Parametru | Valoare |
|---|---|
| Model | `yolov8n-cls.pt` (nano, 2.7M params) |
| imgsz | 224 |
| Batch | 128 |
| lr0 | 0.001 |
| Epoci | 24/100 (early stop) |

| Metrică | Valoare |
|---|---|
| **Top-1 Accuracy** | **76.95%** |
| Top-5 Accuracy | 100% |

**Problemă:** Overfitting sever — train_loss=0.27 vs val_loss=1.29. `batch=128` → doar 7 batch-uri/epocă, insuficient. `imgsz=224` prea mic pentru imagini de 392px.

---

### Trial C2 — Classification nano, imgsz=320, regularizare

**Run:** `runs/classify/runs/classify/basil_health_v2/`

| Parametru | C1 | C2 |
|---|---|---|
| imgsz | 224 | **320** |
| Batch | 128 | **32** |
| lr0 | 0.001 | **0.0003** |
| cos_lr | — | **True** |
| label_smoothing | — | **0.1** |
| dropout | — | **0.2** |
| Epoci | 24 | 54/150 |

| Metrică | Valoare |
|---|---|
| **Top-1 Accuracy** | **78.44%** |
| Top-5 Accuracy | 100% |
| Gap val-train loss | 0.46 (overfitting moderat) |

**Îmbunătățire:** +1.5% vs C1. Val_loss mai stabil (0.54–0.93). Dar plafonul nano persistă — modelul nu are suficientă capacitate.

---

### Trial C3 — Classification small, imgsz=320

**Run:** `runs/classify/runs/classify/basil_health_v3/`

| Parametru | C2 | C3 |
|---|---|---|
| Model | `yolov8n-cls` (2.7M) | **`yolov8s-cls` (6.5M)** |
| Restul | identic | identic |
| Epoci | 54 | 41/150 |

| Metrică | Valoare |
|---|---|
| **Top-1 Accuracy** | **81.04%** |
| Top-5 Accuracy | 100% |
| Gap val-train loss | 0.47 (overfitting moderat) |

**Îmbunătățire:** +2.6% vs C2. Model small prinde mai bine pattern-urile subtile ale bolilor.

**Problemă rămasă:** val_loss oscilant (0.54–0.94). Cauza probabilă: 269 imagini val = set mic → metrici zgomotoase. Overfitting moderat persistent.

**Cel mai bun model actual:** `runs/classify/runs/classify/basil_health_v3/weights/best.pt` — **81.04% Top-1**

---

### Trial C4 — Classification small, imgsz=416, lr=0.0001

**Run:** `runs/classify/runs/classify/basil_health_v4/`

| Parametru | C3 | C4 |
|---|---|---|
| Model | `yolov8s-cls` | `yolov8s-cls` (identic) |
| imgsz | 320 | **416** |
| lr0 | 0.0003 | **0.0001** |
| weight_decay | — | **0.002** |
| dropout | 0.2 | **0.3** |
| patience | — | **15** |
| warmup_epochs | — | **5** |
| Epoci | 41 | 58/200 (early stop ep. 43) |

| Metrică | Valoare |
|---|---|
| **Top-1 Accuracy** | **79.93%** |
| Top-5 Accuracy | 100% |

**Concluzie:** C4 nu a depășit C3. Creșterea `imgsz` la 416 și reducerea `lr0` la 0.0001 au dus la convergență mai lentă, dar platoul a rămas sub C3. Early stopping a oprit antrenamentul la epoca 58 (best = ep. 43). Overfitting-ul moderat persistent — val_loss oscilant din cauza setului mic de validare (269 imagini).

**Cel mai bun model:** `runs/classify/runs/classify/basil_health_v3/weights/best.pt` — **81.04% Top-1** (C3 rămâne campion)

---

## Plan următori pași

### Scurt termen — îmbunătățire model clasificare
- [x] C4: `imgsz=416`, `lr0=0.0001`, `patience=15` — 79.93% (sub C3)
- [ ] Verificare per-clasă: confusion matrix pentru a vedea care clasă e problema

### Termen mediu — detection cu bbox-uri reale
- [ ] Adnotare 300-400 imagini în **Roboflow Annotate** (gratuit) cu bbox-uri reale pe leziuni
- [ ] Reantrenare `yolov8s.pt` (detection) pe datele adnotate → target mAP50 > 0.80

### Deployment
- [ ] Export `best.pt` → ONNX / TFLite pentru Raspberry Pi 5
- [ ] Benchmark inferență pe Pi 5 (target < 100ms/frame)
- [ ] Strat de decizie PID bazat pe output clasificare
- [ ] Test pipeline complet pe hardware real

---

*Ultima actualizare: 2026-06-11 — C4 finalizat (79.93%), C3 rămâne best (81.04%)*
