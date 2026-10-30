# Convenții

## Structură

| Tip de fișier | Locație |
|---|---|
| cod reutilizabil | `src/<modul>/` — `training`, `data`, `edge`, `calibration` |
| utilitare de mediu | `tools/` |
| run-uri de antrenare | `experiments/<classification\|detection>/<nume_run>/` |
| modele pregătite pentru deployment | `models/<nume>.pt` (copie a `best.pt`) |
| documentație | `docs/`, câte un fișier pe capitol din teză |
| date locale generate | `date/`, `config/`, `.local/` (excluse din git) |

## Cod Python

- Python 3.11+, cu adnotări de tip la funcțiile publice.
- Fiecare script are un docstring de modul cu scopul, anexa din teză (dacă există) și un exemplu de rulare.
- CLI cu `argparse`; valori implicite care funcționează fără argumente.
- Căile se calculează din `ROOT = Path(__file__).resolve().parents[2]`.
- Importurile grele (`ultralytics`, `torch`, `cv2`, `gpiozero`) se fac în interiorul funcțiilor, ca `--help` și testele să ruleze fără ele.
- Comentariile și mesajele sunt în română fără diacritice, în stilul anexelor din teză. Identificatorii sunt în engleză.
- Codul hardware are un mod de simulare implicit; accesul la GPIO se face numai cu `--hardware`.
- Clasele modelului se citesc din `model.names`, niciodată din indici ficși.

## Protocol pentru experimente

1. Adaugă un preset nou în `src/training/train_classification.py` (sau rulează cu `--name`). Numele run-ului este `basil_health_v<N>`.
2. Rulează antrenarea. Ultralytics salvează `args.yaml` și `results.csv` în `experiments/classification/<run>/`.
3. Evaluează pe test cu `evaluate_classification.py`, pe **același subset** cu modelul de referință cu care compari.
4. Documentează:
   - un rând nou în `experiments/README.md`;
   - o coloană nouă în tabelul din `docs/model-ai.md` §5, plus 2–3 rânduri de analiză: ce s-a schimbat, de ce, ce a rezultat;
   - dacă noul model devine referință: copiază-l în `models/`, apoi actualizează `models/README.md` și `dev-pack/CONTEXT.md`.
5. Un experiment eșuat se documentează și el. Concluziile negative (C4, D2) sunt rezultate valide.

## Documentație

- Limba este româna, cu diacritice. Termenii tehnici consacrați rămân în engleză.
- Virgulă zecimală în proză (81,04%) și punct în cod și în configurații (0.0003).
- Fiecare cifră are o sursă verificabilă: un fișier din `experiments/` sau o comandă din `docs/reproducibilitate.md`.
- Ce diferă de textul tezei se trece în `docs/rezultate.md` §6 (erate), nu se rescrie în tăcere.
- Diagramele se fac în Mermaid, ca să poată fi editate în text.

## Git

- Mesajele de commit se scriu la imperativ, în română sau engleză, cu un prefix de tip: `docs:`, `feat:`, `fix:`, `exp:`, `refactor:`.
- Weights noi se versionează numai pentru modelele de referință. Celelalte rămân local.
- Nu se versionează: `.venv/`, `*.onnx`, `date/*.db`, `config/*.json`, imagini de test personale.
