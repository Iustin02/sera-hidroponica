# Arhivă — scripturi exploratorii

Aceste scripturi provin din fazele inițiale ale proiectului, de dinainte de alegerea abordării finale. **Nu fac parte din metodologia tezei** și nu sunt întreținute. Sunt păstrate pentru trasabilitate.

| Fișier | Scop inițial |
|---|---|
| `train.py` | Antrenare generică YOLOv8 (CUDA/MPS/CPU, auto-batch) |
| `download_inat_yolo.py` | Descărcarea de imagini iNaturalist pentru detecția prezenței plantei |
| `autolabel_herbs_bbox.py` | Etichetare automată de bbox-uri pe imagini iNaturalist |
| `download_commons_images.py`, `commons_queries.yaml` | Descărcarea de imagini cu licență liberă de pe Wikimedia Commons |
| `init_disease_dataset.py` | Generarea scheletului unui set de date YOLO pentru boli |

Starea completă a depozitului dinaintea reorganizării (prototipurile pe tomate, setul de detecție cu 5 clase, ghidurile de sesiune) este păstrată în tag-ul git [`v0.9-pre-restructure`](https://github.com/Iustin02/sera-hidroponica/tree/v0.9-pre-restructure).
