# Documentație

Documentația tehnică urmează structura lucrării de diplomă.

| Document | Capitol teză | Conținut |
|---|---|---|
| [arhitectura.md](arhitectura.md) | 1, 2, 4.1–4.2 | NFT și homeostazia soluției, cerințe CF/CNF, arhitectura pe 3 niveluri, stiva software, schema SQLite |
| [hardware.md](hardware.md) | 3 | Raspberry Pi 5, senzori, actuatoare, pini GPIO, cablaj, calibrare |
| [model-ai.md](model-ai.md) | 4.3, Anexele B–D, F | setul de date, YOLOv8s-cls, experimentele C1–C5 și D1–D2 |
| [rezultate.md](rezultate.md) | 5, Concluzii | evaluare, decalaj de domeniu, performanța pe Pi, limitări, **erate** |
| [reproducibilitate.md](reproducibilitate.md) | — | comenzile exacte pentru reproducerea fiecărui rezultat și pentru deployment |

Figurile din `figures/training_report/` sunt cele incluse în teză (Fig. 4.1–4.4, Anexa D). Se regenerează din `experiments/` cu `src/training/generate_training_report.py`.

## Convenții

- Limba documentației este româna. Termenii tehnici consacrați rămân în engleză (*batch*, *dropout*, *mAP*).
- Valorile numerice se scriu cu virgulă zecimală în text (81,04%) și cu punct în cod și în tabelele de configurație.
- Orice rezultat nou se trece întâi în `experiments/` (`args.yaml` + `results.csv`), apoi în `model-ai.md`. Valorile se raportează numai după ce se verifică în fișierele sursă.
