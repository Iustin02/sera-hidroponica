# Instrucțiuni GitHub Copilot

Urmează regulile din [`AGENTS.md`](../AGENTS.md) și convențiile din [`dev-pack/CONVENTIONS.md`](../dev-pack/CONVENTIONS.md).

Pe scurt:
- Proiect de diplomă: sistem NFT pentru busuioc, Edge AI pe Raspberry Pi 5, YOLOv8s-cls cu 3 clase.
- Clasele se citesc din `model.names` (0 = fusarium, 1 = healthy, 2 = powdery). Fără indici ficși.
- Căi relative: `ROOT = Path(__file__).resolve().parents[2]`.
- Codul hardware are mod de simulare implicit; GPIO numai cu `--hardware`.
- Nu modifica `experiments/` și `data/`; nu inventa metrici.
