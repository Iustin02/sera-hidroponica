# Prompt — deployment pe Raspberry Pi 5

```
Citește AGENTS.md, docs/reproducibilitate.md §5 și docs/hardware.md.

Context: Raspberry Pi 5, Raspberry Pi OS 64-bit, acces prin SSH (rulez eu comenzile pe Pi).
Model: <models/basil_health_c3.pt | models/basil_health_c5.pt>
Obiectiv: <instalare inițială | actualizare model | export ONNX | benchmark latență>

Cerințe:
- Dă-mi comenzile în ordine, câte un bloc pe pas, cu rezultatul așteptat după fiecare.
- Mediu virtual obligatoriu (PEP 668); pentru pip folosește TMPDIR pe disc (/tmp e tmpfs de 2 GB).
- Orice comandă pe GPIO rulează întâi în --dry-run; trecerea la --hardware o fac eu.
- Pentru benchmark: 50 de cadre de încălzire, apoi media și p95 pe 200 de cadre; raportează versiunile pachetelor.
- Nu pune în fișiere IP-uri, parole sau nume de utilizator.
- La final: propune actualizarea docs/rezultate.md §4 cu cifrele pe care le obțin eu.
```
