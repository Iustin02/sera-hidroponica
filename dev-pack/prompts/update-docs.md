# Prompt — sincronizarea documentației

```
Citește AGENTS.md, docs/README.md și dev-pack/CONVENTIONS.md (secțiunea „Documentație”).

Ce s-a schimbat: <cod / rezultat nou / hardware / nimic: audit general>

Sarcină:
1. Compară fiecare cifră din README.md, docs/*.md, models/README.md și dev-pack/CONTEXT.md cu
   sursa ei (experiments/*/results.csv, args.yaml sau ieșirea evaluate_classification.py).
   Fă un tabel: afirmație | fișier:linie | sursă | corect/greșit.
2. Verifică că toate linkurile relative și ancorele duc la fișiere și secțiuni existente.
3. Verifică tabelul de corespondență teză ↔ depozit din README.md.
4. Propune corecturile ca diff; diferențele față de textul tezei se trec în docs/rezultate.md §6.
5. Nu modifica experiments/ sau data/.
```
