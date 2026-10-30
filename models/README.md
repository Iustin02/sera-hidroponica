# Modele pentru deployment

| Fișier | Experiment | Arhitectură | imgsz | Top-1 val | Top-1 test | Utilizare |
|---|---|---|---|---|---|---|
| `basil_health_c3.pt` | C3 (`basil_health_v3`) | YOLOv8s-cls | 320 | 81,04% | 85,82% | modelul raportat în teză |
| `basil_health_c5.pt` | C5 (`basil_health_v4-2`) | YOLOv8s-cls | 416 | 86,74% | 85,06% | modelul rulat pe Raspberry Pi 5 |

Fișierele sunt copii identice ale `experiments/classification/<run>/weights/best.pt`. Git le stochează o singură dată, ca același obiect.

Clasele, în ordinea indicilor modelului (`model.names`): `0: fusarium`, `1: healthy`, `2: powdery`.

```python
from ultralytics import YOLO
model = YOLO("models/basil_health_c3.pt")
result = model.predict("frunza.jpg", imgsz=320)[0]
print(model.names[result.probs.top1], float(result.probs.top1conf))
```

Exporturile ONNX (`*.onnx`) se generează cu `src/edge/export_onnx.py` și sunt excluse din git.
