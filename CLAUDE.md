@AGENTS.md

## Specific Claude Code

- Comenzi rapide în `.claude/commands/`: `/experiment`, `/deploy-pi`, `/update-docs`, `/review`. Fiecare încarcă promptul corespunzător din `dev-pack/prompts/`.
- Pentru sarcini cu mai mulți pași, propune întâi un plan scurt și așteaptă confirmarea înainte de a modifica `docs/` sau `src/edge/`.
- La final, rulează comenzile de verificare din `AGENTS.md` relevante pentru modificare și raportează rezultatul real.
