# CONTEXT — faptele proiectului

> Sursa de adevăr pentru asistenții AI. Dacă un fapt de aici intră în conflict cu altă sursă, verifică în `experiments/` și actualizează acest fișier.

## Identitate

- **Titlu:** Utilizarea Edge AI în managementul autonom al culturilor NFT
- **Tip:** proiect de diplomă, USV Suceava, FIESC, Automatică și Informatică Aplicată, 2025–2026
- **Autor:** Iustin Iulian Dodiță (GitHub `Iustin02`) · **Coordonator:** Ș.l. dr. ing. Corneliu Buzduga
- **Cultura:** exclusiv busuioc (*Ocimum basilicum*). Prototipurile vechi pe tomate și mentă sunt arhivate și nu se reintroduc.

## Valori de referință (verificate)

| Fapt | Valoare | Sursă |
|---|---|---|
| Model din teză | C3 = `basil_health_v3`, yolov8s-cls, imgsz 320 | `experiments/classification/basil_health_v3/args.yaml` |
| C3 Top-1 val / test | 81,04% (269 img) / 85,82% (134 img) | `evaluate_classification.py --subset roboflow` |
| Cel mai bun model | C5 = `basil_health_v4-2`, yolov8s-cls, imgsz 416 | `experiments/classification/basil_health_v4-2/args.yaml` |
| C5 Top-1 val / test | 86,74% (347) / 85,06% (174) | `evaluate_classification.py --subset all` |
| Pe Raspberry Pi 5 (C5, CPU) | 85,06% Top-1, ~258 ms/cadru | măsurătoare din 2026-06-28 |
| Detecție D1 (bbox generice) | mAP50 val 0,654; teza raportează 45,98% pe test | `experiments/detection/basil_3class/results.csv` |
| Set Roboflow | 1.344 imagini, 941/269/134 | fișierele fără prefixul `hq_frame_` |
| Set extins | 1.736 imagini, 1.215/347/174 | `data/basil_classification/` |
| Ordinea claselor | 0 = fusarium, 1 = healthy, 2 = powdery | `model.names` |

## Decizii și motivele lor

| Decizie | Motiv |
|---|---|
| Clasificare în loc de detecție | fără bbox-uri manuale, detecția are un plafon (mAP50 ≤ 0,65) |
| yolov8s în loc de yolov8n | +2,6 pp la configurație identică (C2 → C3) |
| batch 32, nu 128 | cu 941 de imagini, batch 128 dă 7 iterații pe epocă și overfitting sever (C1) |
| Pompă de recirculare externă | nu încălzește soluția și se reamorsează după o cădere de curent |
| Pantă NFT de 3° | peliculă mai rapidă, aproximativ +5% oxigen dizolvat |
| ADS1115 alimentat la 5V | acoperă ieșirile 0–5V ale modulelor; I2C rămâne la 3,3V |

## Invarianți

- `experiments/**/args.yaml`, `results.csv`, `weights/best.pt` și `data/**` se modifică numai la cerere explicită.
- `models/basil_health_c3.pt` trebuie să dea 81,04% pe `--subset roboflow --split val`. Altfel ceva este stricat.
- Schema SQLite (coloanele `ph`, `ec_ms_cm`, `water_temp_c`, `ambient_temp_c`, `humidity_pct`, `light_lux`, `water_level_pct`, `reading_at`) este cea din teză (Anexa A). Coloanele noi se adaugă numai prin migrare documentată.
- Pinii BCM validați: dozare A = 5/6, dozare B = 13/19, releu ventilator = 16, recirculare PWM = 18, 1-Wire = 4, I2C = 2/3.

## Probleme deschise

- Decalaj de domeniu pe video real: plantele sănătoase sunt clasificate ca *Fusarium*.
- Latența pe Pi (~258 ms) este peste ținta CNF1 de 100 ms; urmează ONNX, imgsz 320 și INT8.
- Cadrele HQ sunt împărțite aleator între split-uri, cu posibilă scurgere de informație. Trebuie refăcută o împărțire pe secvențe.
- Hardware: alimentarea ventilatorului prin releu, detecția DS18B20, integrarea DHT22, BH1750 și a senzorului de nivel.
- Lipsesc adnotări bbox manuale (300–400 de imagini) pentru detecție reală.
