# Checklist după un experiment

- [ ] run salvat în `experiments/<tip>/<nume_run>/`, cu `args.yaml` și `results.csv`
- [ ] o singură variabilă schimbată față de referință (sau abaterea e justificată în documentație)
- [ ] evaluare pe val **și** pe test, pe același subset ca modelul de referință
- [ ] recall per clasă notat, cu atenție la `fusarium → healthy` (eroarea cea mai costisitoare)
- [ ] rând nou în `experiments/README.md`
- [ ] coloană nouă și analiză în `docs/model-ai.md` §5
- [ ] dacă devine referință: copiat în `models/`, apoi actualizate `models/README.md` și `dev-pack/CONTEXT.md`
- [ ] grafice regenerate, dacă e cazul: `python src/training/generate_training_report.py`
