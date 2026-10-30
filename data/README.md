# Seturi de date

## `basil_classification/` — clasificare, 3 clase

Format folder (ImageFolder): `{train,val,test}/{fusarium,healthy,powdery}/*.jpg`.

| Split | fusarium | healthy | powdery | Total |
|---|---|---|---|---|
| train | 321 | 588 | 306 | 1.215 |
| val | 88 | 164 | 95 | 347 |
| test | 39 | 88 | 47 | 174 |
| **Total** | 448 | 840 | 448 | **1.736** |

Componența:

| Subset | Imagini | Proveniență |
|---|---|---|
| Roboflow | 1.344 | *Basil Health Detection*, exportat de pe [Roboflow Universe](https://universe.roboflow.com/) pe 2026-06-11, fără pre-procesare sau augmentare (vezi `README.roboflow.txt`). Fișierele au prefixul `IMG_…rf.<hash>.jpg`. |
| Cadre HQ | 392 | Cadre extrase cu ffmpeg (`-q:v 2`) dintr-o înregistrare video proprie a serei hidroponice (1.072×1.904 px). Toate sunt în clasa `healthy`. Fișierele se numesc `hq_frame_<n>.jpg`. Au fost adăugate pe 2026-06-16, după teză. |

- **Setul din teză (C1–C4)** = subsetul Roboflow: 941 / 269 / 134.
- **Setul extins (C5)** = Roboflow + cadre HQ: 1.215 / 347 / 174.
- `evaluate_classification.py --subset roboflow` exclude automat cadrele HQ.

> **Atenție.** Cadrele HQ provin dintr-un singur video și au fost împărțite aleator între split-uri. Cadrele vecine sunt aproape identice, deci există un risc de scurgere de informație între train și test pentru clasa `healthy` (vezi [`docs/rezultate.md` §5](../docs/rezultate.md#5-limitări-și-amenințări-la-validitate)).

## `basil_detection_3class/` — detecție, 3 clase

Este generat din setul de clasificare de [`src/data/build_detection_dataset.py`](../src/data/build_detection_dataset.py). Format YOLO: `images/{split}/*.jpg` + `labels/{split}/*.txt`. Fiecare etichetă este un **bbox generic centrat** `<cls> 0.5 0.5 0.85 0.85`, nu o adnotare manuală a leziunii. Ordinea claselor: 0 = healthy, 1 = Fusarium oxysporum, 2 = Oidium belbahrii.

## Licențe

- Imaginile Roboflow rămân sub licența publicată de autorii setului *Basil Health Detection* pe Roboflow Universe. Înainte de redistribuire, verifică termenii de pe pagina setului.
- Cadrele `hq_frame_*` aparțin autorului proiectului și sunt distribuite împreună cu codul.
