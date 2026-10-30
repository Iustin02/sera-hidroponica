# Checklist înainte de commit

- [ ] `python -m compileall -q src tools` trece fără erori
- [ ] `git diff --stat` nu conține modificări neintenționate în `experiments/`, `data/` sau `models/`
- [ ] nu există căi absolute: `git grep -n "/Users/\|/home/" -- src tools` nu întoarce nimic
- [ ] nu există secrete: `git grep -n -i "password\|passwd\|api_key\|192\.168\." -- . ":!*.jpg" ":!dev-pack"` nu întoarce nimic
- [ ] nu sunt adăugate fișiere generate: `*.db`, `*.onnx`, `config/*.json`, `.venv/`
- [ ] codul hardware rulează în `--dry-run`: `python src/edge/control_loop.py --once`
- [ ] dacă s-a atins modelul sau evaluarea, C3 dă încă 81,04% pe `--subset roboflow --split val`
- [ ] documentația e sincronizată (`docs/`, `experiments/README.md`, `dev-pack/CONTEXT.md`)
- [ ] mesajul de commit are prefix (`docs:`, `feat:`, `fix:`, `exp:`, `refactor:`)
