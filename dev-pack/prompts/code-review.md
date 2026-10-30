# Prompt — review critic

```
Citește AGENTS.md și dev-pack/CONVENTIONS.md, apoi fă un review al modificărilor curente (git diff).

Caută, în această ordine:
1. Corectitudine: indici de clasă ficși în loc de model.names, coloane SQL inexistente în schemă,
   unități greșite (µS/cm vs mS/cm), pini în conflict, pași de calibrare cu semn greșit.
2. Siguranța hardware: GPIO activat fără --hardware, pompe care pot rămâne pornite la o excepție
   (lipsă try/finally), mai multe procese care citesc ADS1115.
3. Reproductibilitate: căi absolute, seed lipsă, metrici comparate pe subseturi diferite.
4. Date și secrete: modificări în experiments/ sau data/, IP-uri, parole, fișiere .db/.json de calibrare.
5. Documentație: cifre noi fără sursă, docs/ nesincronizat cu codul.

Pentru fiecare problemă: fișier:linie, scenariul concret în care apare, corectura propusă.
Nu raporta preferințe de stil decât dacă încalcă CONVENTIONS.md.
```
