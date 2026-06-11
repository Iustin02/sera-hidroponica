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
| `Downloads/Basil Health Detection/` | Classification (foldere) | 3 clase | identic cu cel din proiect |

**Clase în `basil_dataset` (5 clase — Roboflow):**
- `healthy`
- `downy_mildew`
- `bacterial_leaf_spot`
- `leaf_miner_damage`
- `general_disease`

**Clase în `basil_classification_dataset` (3 clase — Roboflow):**
- `healthy`
- `fusarium`
- `powdery`

### Model existent la start

Script: `train_basil_complete.py` → `yolov8n.pt` pe `basil_dataset` (5 clase)

Antrenament: **3 epoci** (s-a oprit manual sau prin eroare)

| Metrică | Valoare |
|---|---|
| mAP50 | 0.076 (epoca 1) |
| Precision | 0.666 |
| Recall | 0.199 |

**Concluzie:** model nefuncțional, epoci insuficiente, clase nealiniate cu obiectivul proiectului.

---

## Decizie arhitecturală — schema finală de 3 clase

**Motivație:** 5 clase sunt prea multe pentru un model nano pe Raspberry Pi 5, și nu se aliniază cu logica de control PID (healthy / disease / nutrient issue).

**Schema aleasă** (fișier `BASIL_DETECTION_CLASS_SCHEMA.md`):

| ID | Clasă | Denumire simplă |
|---|---|---|
| 0 | `healthy` | sănătoasă |
| 1 | `Fusarium oxysporum` | fusarioză |
| 2 | `Oidium belbahrii` | făinare |

**De ce aceste 2 boli:** sunt cele mai frecvente la busuioc în sisteme hidroponic și există imagini reale în dataset-ul descărcat (`fusarium` și `powdery` din `basil_classification_dataset`).

**Mapping control:**
- `healthy` → nicio acțiune
- `Fusarium oxysporum` → alertă + izolare + verificare pH/EC
- `Oidium belbahrii` → creștere ventilație, scădere umiditate sub 60%

---

## Trial 0 — Conversie dataset classification → detection

**Fișier:** `build_detection_dataset.py`
**Output:** `basil_detection_3class/`

**Ce s-a făcut:** dataset-ul de clasificare (foldere pe clasă, fără bbox) a fost convertit în format YOLO detection prin generarea de bounding box-uri care acoperă întreaga imagine.

| Parametru | Valoare |
|---|---|
| Box size | `0.5 0.5 1.0 1.0` (full-image) |
| Train | 941 imagini |
| Val | 269 imagini |
| Test | 134 imagini |

**Problema identificată ulterior:** box-urile 1.0×1.0 acoperă perfect toată imaginea. Combinate cu mosaic augmentation (care unește 4 imagini), modelul învață să detecteze simultan mai multe boli pe aceeași imagine. La inferență pe o singură imagine prezice multiple box-uri suprapuse → NMS ineficient → precision scăzut.

---

## Trial 1 — Primul antrenament (v1)

**Script:** `train_detection_3class.py`
**Run:** `runs/detect/runs/detect/basil_3class/`

**Configurație:**

| Parametru | Valoare |
|---|---|
| Model | `yolov8n.pt` (detection) |
| Dataset | `basil_detection_3class/` (3 clase) |
| Epoci | 100 (early stop la 25) |
| Patience | 5 |
| imgsz | 416 |
| Batch | 16 |
| Device | MPS (Apple M1 Pro) |
| Mosaic | activat (default 1.0) |
| Box loss weight | 7.5 (default) |
| Cls loss weight | 0.5 (default) |
| Box size în dataset | 1.0×1.0 (full-image) |

**Rezultate — cea mai bună epocă (20/25):**

| Metrică | Valoare |
|---|---|
| mAP50 | **0.654** |
| mAP50-95 | 0.624 |
| Precision | 0.477 ❌ |
| Recall | 0.861 ✅ |

**Problema:** Recall bun (modelul găsește boala), dar Precision scăzut (prea multe false positives). Cauza: mosaic + full-image boxes.

**Early stop:** La epoca 25 (fără îmbunătățiri față de epoca 20).

---

## Trial 2 — Fix precision (v2) — în curs

**Script:** `train_detection_3class.py`
**Run:** `runs/detect/basil_3class_v2/`

**Modificări față de Trial 1:**

| Parametru | Trial 1 | Trial 2 | Motiv |
|---|---|---|---|
| Box size | 1.0×1.0 | **0.85×0.85** | reduce suprapunerea perfectă a box-urilor |
| Mosaic | 1.0 (activ) | **0.0 (dezactivat)** | mosaic cu full-image boxes confundă detectorul |
| Box loss weight | 7.5 | **3.0** | localizarea e mai puțin importantă decât clasificarea |
| Cls loss weight | 0.5 | **1.5** | accent pe clasificarea corectă a bolii |

**Status:** antrenament în desfășurare.

**Așteptări:** Precision > 0.70, mAP50 > 0.70

---

## To-Do următori pași

- [ ] `train-pi-model` — exportare model optimizat pentru Raspberry Pi 5 (TFLite / ONNX)
- [ ] `optimize-inference` — benchmark inferență pe Pi 5, target < 100ms/frame
- [ ] `build-control-logic` — strat de decizie PID bazat pe output-ul modelului
- [ ] `test-on-pi` — testare pipeline complet pe hardware real

---

*Ultima actualizare: 2026-06-11*
