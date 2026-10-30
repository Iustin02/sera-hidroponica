# Prompt — experiment nou de antrenare

```
Citește AGENTS.md, dev-pack/CONTEXT.md și dev-pack/CONVENTIONS.md (secțiunea „Protocol pentru experimente”).

Obiectiv: <ce vreau să îmbunătățesc, de ex. recall-ul pe fusarium>
Ipoteză: <ce schimbare cred că ajută și de ce>
Model de referință: <C3 | C5>, evaluat pe subsetul <roboflow | all>

Pași:
1. Propune UN singur preset nou în src/training/train_classification.py, care diferă de referință
   printr-o singură variabilă (sau justifică de ce schimbi mai multe).
2. Arată-mi comanda de antrenare și estimează durata. Nu porni antrenarea până nu confirm.
3. După antrenare: rulează evaluate_classification.py pe val și pe test, pe același subset ca referința.
4. Compară cu referința într-un tabel: Top-1 val, Top-1 test, recall per clasă.
5. Actualizează experiments/README.md și docs/model-ai.md §5, inclusiv dacă rezultatul e negativ.
6. Nu suprascrie models/ și nu face commit fără confirmarea mea.
```
