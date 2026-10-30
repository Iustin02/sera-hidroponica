# AGENTS.md

Instrucțiuni pentru asistenți AI de programare (Claude Code, Codex, Cursor, Copilot, Aider). Citește acest fișier înainte de orice modificare.

## Proiectul în 5 rânduri

- Proiect de diplomă (USV, 2025–2026): sistem hidroponic NFT autonom pentru **busuioc**, cu Edge AI pe **Raspberry Pi 5**.
- Clasificator **YOLOv8s-cls**, 3 clase: `fusarium`, `healthy`, `powdery`, cu indicii 0/1/2 în ordine alfabetică. Folosește mereu `model.names`.
- Modelul din teză este **C3** (`models/basil_health_c3.pt`, imgsz 320). Cel mai bun model ulterior este **C5** (`models/basil_health_c5.pt`, imgsz 416).
- Senzori pe ADS1115/I2C, 1-Wire și GPIO; date în SQLite; buclă de control cu histerezis.
- Documentația urmează capitolele tezei; `README.md` are tabelul de corespondență teză ↔ cod.

## Înainte să începi

1. Citește [`dev-pack/CONTEXT.md`](dev-pack/CONTEXT.md): valorile de referință și invarianții care nu se modifică.
2. Citește [`dev-pack/CONVENTIONS.md`](dev-pack/CONVENTIONS.md): stilul de cod, structura și protocolul pentru experimente.
3. Pentru sarcinile frecvente, folosește promptul potrivit din [`dev-pack/prompts/`](dev-pack/prompts/).

## Reguli obligatorii

- **Nu modifica** `experiments/*/args.yaml`, `experiments/*/results.csv` și `data/**` fără o cerere explicită. Ele sunt înregistrările pe care se sprijină rezultatele tezei.
- **Nu inventa cifre.** Orice metrică trecută în documentație se verifică întâi în `results.csv` sau se obține rulând `src/training/evaluate_classification.py`.
- **Nu face commit sau push** fără cererea explicită a autorului.
- **Fără secrete în depozit:** parole, IP-uri, chei API, fișiere `config/*.json` de calibrare, baze de date.
- **Căi relative la depozit:** `ROOT = Path(__file__).resolve().parents[2]`. Nicio cale absolută de tipul `/Users/...`.
- Un singur proces citește ADS1115 (coliziuni I2C), iar pH-ul se citește cu pompele oprite.
- Codul pentru hardware rulează implicit în **simulare** (`--dry-run`). Comanda reală pe GPIO se activează explicit.

## Comenzi de verificare

```bash
python -m compileall -q src tools                       # sintaxă
python src/data/verify_classification_dataset.py        # integritatea setului de date
python src/training/evaluate_classification.py --weights models/basil_health_c3.pt --imgsz 320 --subset roboflow --split val   # trebuie: 81.04%
python src/edge/control_loop.py --once                  # bucla de control în simulare
```

## Unde se află lucrurile

| Ai nevoie de… | Mergi la… |
|---|---|
| arhitectură, cerințe, schema bazei de date | `docs/arhitectura.md` |
| pini, cablaj, calibrare | `docs/hardware.md` |
| hiperparametri și istoricul experimentelor | `docs/model-ai.md`, `experiments/README.md` |
| metrici, limitări, erate față de teză | `docs/rezultate.md` |
| comenzi de reproducere și deployment | `docs/reproducibilitate.md` |
