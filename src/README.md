# Cod sursă

| Modul | Fișier | Anexa din teză | Descriere |
|---|---|---|---|
| `training/` | `train_classification.py` | C | Antrenare YOLOv8-cls; presetări C1–C5 identice cu `args.yaml` |
| | `train_detection.py` | — | Antrenare detector YOLOv8 (presetări D1, D2) |
| | `evaluate_classification.py` | D | Top-1/Top-5, matrice de confuzie, recall per clasă; `--subset roboflow\|all` |
| | `generate_training_report.py` | F | Figurile 4.1–4.4 (netezire EMA, α = 0,3) |
| `data/` | `build_detection_dataset.py` | — | Conversie clasificare → detecție YOLO (bbox generice) |
| | `verify_classification_dataset.py` | — | Verifică structura și numărul de imagini |
| `edge/` | `inference_rpi5.py` | E | Inferență în timp real (cameră USB sau imagine) și decizie de control |
| | `database.py` | A, G | Schema SQLite, inserare și citire |
| | `control_loop.py` | I | Buclă de control cu histerezis; `--dry-run` (implicit) sau `--hardware` |
| | `export_onnx.py` | H | Export ONNX opset 12 |
| `calibration/` | `calibrate_ph.py` | J | Calibrare pH-4502C în 3 puncte |
| | `calibrate_ec.py` | K | Calibrare EC cu etalon de 1413 µS/cm |

Scripturile se rulează din rădăcina depozitului (`python src/<modul>/<fișier>.py`). Fiecare își calculează căile față de propria locație (`ROOT = Path(__file__).resolve().parents[2]`) și nu depinde de directorul curent.

Diferențele față de codul din anexele tezei sunt documentate în [`docs/rezultate.md` §6](../docs/rezultate.md#6-erate-față-de-textul-tezei).
