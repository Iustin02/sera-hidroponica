# CLAUDE.md — Sera Hidroponică / Edge AI NFT (cod & model)

Fișier de context pentru Claude Code. Acest repo conține **codul, dataseturile și modelele AI**
pentru proiectul de licență „Utilizarea Edge AI în managementul autonom al culturilor NFT".
Documentul tezei (`.docx`) este gestionat separat în `~/LICENTA/`.

---

## Despre proiect

- **Scop:** detecția automată a bolilor la **busuioc (Ocimum basilicum)** într-un sistem hidroponic NFT,
  rulând inferență la marginea rețelei (edge) pe **Raspberry Pi 5**.
- **Cultură:** **EXCLUSIV busuioc** — nicio referință la mentă sau roșii (cod legacy de roșii a fost eliminat).
- **Student:** Iustin Iulian Dodiță, GitHub `Iustin02`.

---

## Modelul de clasificare — starea curentă

3 clase: **`healthy`** (588 train) · **`fusarium`** (321 train) · **`powdery`** / Oidium (306 train).

| Exp. | Run | Model | imgsz | Top-1 | Epoci | Note |
|------|-----|-------|-------|-------|-------|------|
| C1 | `basil_health_v1` | yolov8n-cls | 224 | 76.95% | 24 | overfitting |
| C2 | `basil_health_v2` | yolov8n-cls | 320 | 78.44% | 54 | + regularizare |
| C3 | `basil_health_v3` | yolov8s-cls | 320 | 81.04% | 41 | citat în teză (Cap. 6) |
| C4 | `basil_health_v4` | yolov8s-cls | 416 | 79.93% | 58 | sub C3 |
| **C5 ★** | **`basil_health_v4-2`** | **yolov8s-cls** | **416** | **86.74%** | **97** | **best actual** |
| C6 | `basil_health_v5` | yolov8s-cls | 416 | 59% | 2 | abandonat |

- **Cel mai bun model:** `runs/classify/runs/classify/basil_health_v4-2/weights/best.pt` — **86.74% Top-1** (best epoch).
- ⚠️ **Discrepanță de raportare:** teza (Cap. 6) citează încă **C3 (81.04%)**. C5 (86.74%) e ulterior și
  nedocumentat în lucrare — de decis dacă se actualizează narațiunea sau se păstrează C3 ca rezultat raportat.
- Istoric complet și decizii de design: vezi [`MODEL_HISTORY.md`](MODEL_HISTORY.md).

### Limitare cunoscută — decalaj de domeniu (domain shift)
Modelul prezice fals `fusarium` cu încredere mare pe **cadre din video real** (canalul G mai scăzut cu ~29px
față de dataset; compresia video agravează diferența). Vezi `MODEL_HISTORY.md` / teza Cap. 6.5.

---

## Structura repo-ului

| Cale | Conținut |
|------|----------|
| `train_basil_classification.py` | Script principal antrenare clasificator (yolov8s-cls, imgsz=416) |
| `inference.py` | Inferență pe imagine/folder |
| `generate_training_report.py` | Grafice EMA + tabele din `results.csv` (figuri teză) |
| `basil_classification_dataset/` | Dataset clasificare 3 clase (train/val/test) |
| `basil_detection_3class/` | Dataset detecție 3 clase (bbox YOLO) |
| `runs/classify/.../basil_health_v*/` | Run-uri antrenare clasificare (weights + results.csv) |
| `runs/detect/...` | Run-uri detecție (mAP50 ≈ 46% cu bbox generice) |
| `docs/training_report/` | PNG-uri embedate în teză |
| `*.md` (START_HERE, QUICK_START, DOCUMENTATION...) | Ghiduri și jurnale de sesiune |

---

## Mediu de rulare

- **Python:** folosește `python3` sistem (Python 3.14.x cu torch 2.12 + ultralytics 8.4.x).
  Venv-ul local (`venv/`) are `libtorch_cpu.dylib` lipsă — **nu-l folosi**.
- **Mac (dev):** antrenare pe MPS (Apple Silicon M1).
- **Raspberry Pi 5 (deploy):** inferență CPU/ONNX; vezi `requirements.txt` și ghidul de deployment.
- Dependențe: `requirements.txt`.

---

## Convenții & reguli de lucru

- **Busuioc only** — nu reintroduce referințe la mentă/roșii.
- **Nu face commit fără cerere explicită** a utilizatorului.
- **Greutățile modelelor** (`*.pt`) sunt în general ignorate de `.gitignore`; `best.pt`-urile de run sunt
  versionate forțat când e nevoie. Modelele pretrained (`yolov8*.pt` la rădăcină) se descarcă automat.
- Numerotare run-uri: folderul fizic e sursa de adevăr (`basil_health_v4-2` ≠ `v4`). Atenție la confuzii
  C4/C5 — vezi tabelul de mai sus.
- La fiecare run nou de model: actualizează `MODEL_HISTORY.md` (config + best Top-1 + concluzie).

---

## Pași următori (deployment & îmbunătățire)

- [ ] Export `basil_health_v4-2/best.pt` → **ONNX / TFLite** pentru Raspberry Pi 5.
- [ ] Benchmark inferență pe Pi 5 (target < 100 ms/cadru).
- [ ] Reducere domain shift: augmentare cromatică / fine-tune pe cadre video reale.
- [ ] Detecție cu bbox-uri reale adnotate (Roboflow, ~300–400 imagini) → mAP50 > 0.80.
- [ ] Strat de decizie (PID) bazat pe output clasificare → control pompe/ventilator.
- [ ] Decizie teză: actualizare Cap. 6 la C5 (86.74%) sau păstrare C3.

---

*Ultima actualizare: 2026-06-28*
