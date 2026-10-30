# Prompt — integrare senzor sau actuator nou

```
Citește AGENTS.md, dev-pack/CONTEXT.md (secțiunea pini) și docs/hardware.md.

Componentă: <de ex. DHT22 / BH1750 / senzor nivel / ventilator PWM>
Interfață: <GPIO / I2C adresa 0x.. / 1-Wire / analogic pe ADS1115 canal A2-A3>
Pin propus: <BCM nn>, verifică să nu fie deja alocat

Pași:
1. Verifică conflictele de pini și de adrese I2C față de tabelul din docs/hardware.md §4.
2. Scrie modulul de citire în src/edge/ cu mod de simulare (valori plauzibile) când hardware-ul lipsește.
3. Integrează valoarea în database.insert_reading(); dacă e nevoie de o coloană nouă, propune
   migrarea SQL separat și așteaptă confirmarea (schema e cea din teză).
4. Dacă senzorul influențează controlul, extinde control_cycle() cu prag și histerezis documentate.
5. Actualizează docs/hardware.md (tabele 2–4, cablaj) și dev-pack/CONTEXT.md (pini).
6. Verifică: python -m compileall -q src && python src/edge/control_loop.py --once
```
