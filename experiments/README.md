# Experimente

Fiecare folder este un run Ultralytics salvat integral. Fișierele `args.yaml` și `results.csv` sunt **sursa de adevăr** pentru valorile din teză și din documentație.

| Folder | Experiment | Model | Top-1 / mAP50 (val, cel mai bun) | Observații |
|---|---|---|---|---|
| `classification/basil_health_v1` | C1 | yolov8n-cls, 224 | 76,95% | |
| `classification/basil_health_v2` | C2 | yolov8n-cls, 320 | 78,44% | |
| `classification/basil_health_v3` | **C3** | yolov8s-cls, 320 | **81,04%** | modelul din teză → `models/basil_health_c3.pt` |
| `classification/basil_health_v4` | C4 | yolov8s-cls, 416 | 79,93% | |
| `classification/basil_health_v4-2` | **C5** | yolov8s-cls, 416 | **86,74%** | set extins → `models/basil_health_c5.pt` |
| `classification/basil_health_v5` | — | yolov8s-cls, 416 | 59,08% | oprit după 2 epoci; fără weights |
| `detection/basil_3class` | D1 | yolov8n | 0,654 | bbox generice |
| `detection/basil_3class_v2` | D2 | yolov8n | 0,423 | mosaic = 0 |
| `detection/basil_3class_v2-2` | D2′ | yolov8n | 0,443 | repetare D2 |

Conținutul unui run:

```
<run>/
├── args.yaml                       # configurația completă de antrenare
├── results.csv                     # metrici per epocă
├── results.png                     # curbele de antrenare
├── confusion_matrix*.png           # pe setul de validare, la cea mai bună epocă
├── train_batch*.jpg, val_batch*    # mostre vizuale
└── weights/best.pt                 # ponderile celei mai bune epoci
```

> Căile absolute din `args.yaml` (`data:`, `save_dir:`) indică locația de pe mașina de antrenare. Sunt păstrate neschimbate, ca înregistrare istorică.

Numerotarea respectă numele fizice ale folderelor. `basil_health_v4-2` este C5, nu C4.
