# Reproducibilitate

Acest document descrie pașii prin care se reproduce fiecare rezultat din teză și din [`rezultate.md`](rezultate.md). Toate comenzile se rulează din rădăcina depozitului.

## 1. Mediul de lucru

| | Stație de dezvoltare (antrenare) | Raspberry Pi 5 (deployment) |
|---|---|---|
| Python | 3.11+ (testat pe 3.14) | 3.11+ (venv obligatoriu, PEP 668) |
| Dependențe | `requirements.txt` | `requirements-edge.txt` |
| Accelerare | CUDA / Apple MPS / CPU (autodetectată) | CPU |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/setup_check.py      # verifică Python, PyTorch, dispozitivul, Ultralytics
python tools/check_mps.py        # opțional, doar pe Apple Silicon
```

Determinism: toate run-urile folosesc `seed=0` și `deterministic=True`. Pe MPS, unele operații rămân nedeterministe, așa că reantrenarea poate da diferențe de ordinul a ±1 pp. Evaluarea pe CPU a ponderilor salvate este însă exact reproductibilă.

## 2. Verificarea setului de date

```bash
python src/data/verify_classification_dataset.py
```

Rezultat așteptat: train 1.215 · val 347 · test 174 imagini, adică setul extins. Setul Roboflow din teză este același set fără fișierele `hq_frame_*`. Scripturile de evaluare îl construiesc automat cu `--subset roboflow`.

## 3. Reproducerea rezultatelor raportate

| Rezultat | Comandă |
|---|---|
| C3, val — 81,04% (teză) | `python src/training/evaluate_classification.py --weights models/basil_health_c3.pt --imgsz 320 --subset roboflow --split val` |
| C3, test — 85,82% | `python src/training/evaluate_classification.py --weights models/basil_health_c3.pt --imgsz 320 --subset roboflow --split test` |
| C5, val — 86,74% | `python src/training/evaluate_classification.py --weights models/basil_health_c5.pt --imgsz 416 --subset all --split val` |
| C5, test — 85,06% | `python src/training/evaluate_classification.py --weights models/basil_health_c5.pt --imgsz 416 --subset all --split test` |
| Figurile 4.1–4.4 | `python src/training/generate_training_report.py` |

## 4. Reantrenare

```bash
python src/training/train_classification.py --preset C3          # experimentul din teză
python src/training/train_classification.py --preset C5 --name basil_health_v6
```

> C1–C4 au fost antrenate pe setul Roboflow, înainte de adăugarea cadrelor HQ. Pentru o reproducere strictă, mută temporar fișierele `hq_frame_*` în afara `data/basil_classification/`.

Detecție:

```bash
python src/data/build_detection_dataset.py    # regenerează data/basil_detection_3class din setul de clasificare
python src/training/train_detection.py --preset D1
```

Rezultatele se salvează în `experiments/<tip>/<nume_run>/`. Protocolul de documentare a unui run nou este în [`dev-pack/CONVENTIONS.md`](../dev-pack/CONVENTIONS.md#protocol-pentru-experimente).

## 5. Deployment pe Raspberry Pi 5

```bash
# pe Pi (Raspberry Pi OS 64-bit)
sudo apt install -y python3-venv python3-opencv i2c-tools
git clone https://github.com/Iustin02/sera-hidroponica.git && cd sera-hidroponica
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
TMPDIR=~/pip-tmp pip install -r requirements-edge.txt   # /tmp e tmpfs de 2 GB, prea mic pentru torch

python src/edge/database.py                                   # creează date/sera_hidroponica.db
python src/edge/inference_rpi5.py --weights models/basil_health_c5.pt --imgsz 416
python src/edge/control_loop.py --once                        # simulare (dry-run)
python src/edge/control_loop.py --hardware                    # comandă GPIO reală
```

Export ONNX (pe stația de dezvoltare sau pe Pi):

```bash
python src/edge/export_onnx.py --weights models/basil_health_c5.pt --imgsz 416
```

Calibrarea senzorilor (interactivă, cu soluții tampon):

```bash
python src/calibration/calibrate_ph.py
python src/calibration/calibrate_ec.py
```

## 6. Versiuni folosite în teză

| Pachet | Versiune |
|---|---|
| ultralytics | 8.4.66 |
| torch | 2.12.0 |
| Python | 3.14.3 (macOS, M1 Pro) |
