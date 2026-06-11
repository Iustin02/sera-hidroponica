#!/usr/bin/env python3
"""
Generare grafice si tabele pentru documentatia proiectului.
Citeste toate run-urile de clasificare si produce:
  1. Comparatie Top-1 Accuracy (toate run-urile pe acelasi grafic)
  2. Train loss vs Val loss per run
  3. Bar chart - cel mai bun rezultat per run
  4. Tabel rezumat hyperparametri + rezultate
  5. Grafic evolutie val_loss (stabilitate)
Output: docs/training_report/
"""

import csv
import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import numpy as np
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "numpy", "--break-system-packages", "-q"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

BASE = Path("/Users/iustiniuliandodita/sera-hidroponica/runs/classify/runs/classify")
OUT  = Path("/Users/iustiniuliandodita/sera-hidroponica/docs/training_report")
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "C1 — nano / imgsz=224": {
        "csv":    BASE / "basil_health_v1/results.csv",
        "color":  "#e74c3c",
        "params": {"Model": "yolov8n-cls", "imgsz": 224, "batch": 128,
                   "lr0": 0.001, "dropout": "—", "label_smooth": "—"},
    },
    "C2 — nano / imgsz=320": {
        "csv":    BASE / "basil_health_v2/results.csv",
        "color":  "#f39c12",
        "params": {"Model": "yolov8n-cls", "imgsz": 320, "batch": 32,
                   "lr0": 0.0003, "dropout": 0.2, "label_smooth": 0.1},
    },
    "C3 — small / imgsz=320": {
        "csv":    BASE / "basil_health_v3/results.csv",
        "color":  "#2ecc71",
        "params": {"Model": "yolov8s-cls", "imgsz": 320, "batch": 32,
                   "lr0": 0.0003, "dropout": 0.2, "label_smooth": 0.1},
    },
    "C4 — small / imgsz=416": {
        "csv":    BASE / "basil_health_v4/results.csv",
        "color":  "#3498db",
        "params": {"Model": "yolov8s-cls", "imgsz": 416, "batch": 32,
                   "lr0": 0.0001, "dropout": 0.3, "label_smooth": 0.1},
    },
}

# ── Smoothing ────────────────────────────────────────────────────────────────

def ema(values, alpha=0.3):
    """Exponential Moving Average — acelasi algoritm ca TensorBoard smoothing."""
    smoothed, last = [], values[0]
    for v in values:
        last = alpha * v + (1 - alpha) * last
        smoothed.append(last)
    return smoothed

SMOOTH = 0.3   # 0=fara smoothing, 1=linie dreapta; 0.3 e echilibrul ideal

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "text.color":       "#e6edf3",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "monospace",
}
plt.rcParams.update(STYLE)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load(csv_path):
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    keys  = list(rows[0].keys())
    top1  = [k for k in keys if "top1" in k][0]
    trloss= [k for k in keys if "train" in k and "loss" in k][0]
    vlloss= [k for k in keys if "val" in k and "loss" in k][0]
    epochs     = [int(r[keys[0]].strip()) for r in rows]
    top1_vals  = [float(r[top1]) * 100 for r in rows]
    train_loss = [float(r[trloss]) for r in rows]
    val_loss   = [float(r[vlloss]) for r in rows]
    return {"epochs": epochs, "top1": top1_vals,
            "train_loss": train_loss, "val_loss": val_loss}

data = {name: load(cfg["csv"]) for name, cfg in RUNS.items()}
available = {k: v for k, v in data.items() if v is not None}

print(f"Run-uri gasite: {list(available.keys())}")

# ── 1. Comparatie Top-1 Accuracy ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Evolutie Top-1 Accuracy — toate run-urile", fontsize=14, fontweight="bold", y=1.01)

for name, d in available.items():
    color = RUNS[name]["color"]
    smooth = ema(d["top1"], SMOOTH)
    # linie bruta — subtila in fundal
    ax.plot(d["epochs"], d["top1"], color=color, linewidth=0.8, alpha=0.2)
    # linie netezita — principala
    ax.plot(d["epochs"], smooth, color=color, linewidth=2.5, label=name)
    best_idx = int(np.argmax(smooth))
    ax.scatter(d["epochs"][best_idx], smooth[best_idx],
               color=color, s=90, zorder=5)
    ax.annotate(f'{max(d["top1"]):.1f}%',
                xy=(d["epochs"][best_idx], smooth[best_idx]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8.5, color=color, fontweight="bold")

ax.set_xlabel("Epoca", fontsize=11)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
ax.set_ylim(30, 100)
ax.axhline(y=80, color="#8b949e", linestyle="--", linewidth=0.8, alpha=0.5, label="80% prag")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / "1_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Salvat: 1_accuracy_comparison.png")

# ── 2. Train vs Val Loss per run ─────────────────────────────────────────────

n = len(available)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
if n == 1:
    axes = [axes]
fig.suptitle("Train Loss vs Val Loss — per run", fontsize=13, fontweight="bold")

for ax, (name, d) in zip(axes, available.items()):
    color = RUNS[name]["color"]
    tr_s = ema(d["train_loss"], SMOOTH)
    vl_s = ema(d["val_loss"],   SMOOTH)
    # brut in fundal
    ax.plot(d["epochs"], d["train_loss"], color=color, linewidth=0.7, alpha=0.18)
    ax.plot(d["epochs"], d["val_loss"],   color=color, linewidth=0.7, alpha=0.18, linestyle="--")
    # neted principal
    ax.plot(d["epochs"], tr_s, color=color, linewidth=2,   label="Train loss")
    ax.plot(d["epochs"], vl_s, color=color, linewidth=2,
            linestyle="--", alpha=0.75, label="Val loss")
    best_acc = max(d["top1"])
    ax.set_title(f"{name}\nBest: {best_acc:.1f}%", fontsize=9, fontweight="bold")
    ax.set_xlabel("Epoca", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(OUT / "2_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Salvat: 2_loss_curves.png")

# ── 3. Bar chart — cel mai bun rezultat per run ───────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Best Top-1 Accuracy per Run", fontsize=13, fontweight="bold")

names  = list(available.keys())
bests  = [max(d["top1"]) for d in available.values()]
colors = [RUNS[n]["color"] for n in names]
epochs = [available[n]["epochs"][int(np.argmax(available[n]["top1"]))] for n in names]

bars = ax.bar(names, bests, color=colors, width=0.5, edgecolor="#30363d", linewidth=1.2)

for bar, val, ep in zip(bars, bests, epochs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.2f}%\n(ep. {ep})", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#e6edf3")

ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.axhline(y=80, color="#8b949e", linestyle="--", linewidth=1, alpha=0.6)
ax.set_xticklabels(names, fontsize=8, rotation=10, ha="right")
ax.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / "3_best_accuracy_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Salvat: 3_best_accuracy_bar.png")

# ── 4. Tabel rezumat ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(18, len(available) * 1.1 + 2))
ax.axis("off")
fig.suptitle("Rezumat Hyperparametri & Rezultate", fontsize=13, fontweight="bold")

headers = ["Run", "Model", "imgsz", "Batch", "lr0", "Dropout", "Label\nSmooth",
           "Epoci\nrulate", "Best\nEpoca", "Best\nTop-1 %"]

# largimi relative per coloana (suma = 1.0)
col_widths = [0.26, 0.12, 0.07, 0.07, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]

rows_tbl = []
for name, d in available.items():
    p = RUNS[name]["params"]
    best_idx = int(np.argmax(d["top1"]))
    rows_tbl.append([
        name,
        p["Model"],
        str(p["imgsz"]),
        str(p["batch"]),
        str(p["lr0"]),
        str(p["dropout"]),
        str(p["label_smooth"]),
        str(len(d["epochs"])),
        str(d["epochs"][best_idx]),
        f'{d["top1"][best_idx]:.2f}%',
    ])

tbl = ax.table(cellText=rows_tbl, colLabels=headers,
               loc="center", cellLoc="center",
               colWidths=col_widths)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.2)

# Color header
for j in range(len(headers)):
    tbl[0, j].set_facecolor("#1f6feb")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Color best row
best_run_idx = int(np.argmax([max(d["top1"]) for d in available.values()]))
for j in range(len(headers)):
    tbl[best_run_idx + 1, j].set_facecolor("#1a4731")
    tbl[best_run_idx + 1, j].set_text_props(color="#e6edf3", fontweight="bold")

# Alternating rows
for i, _ in enumerate(rows_tbl):
    if i != best_run_idx:
        bg = "#161b22" if i % 2 == 0 else "#1c2128"
        for j in range(len(headers)):
            tbl[i + 1, j].set_facecolor(bg)
            tbl[i + 1, j].set_text_props(color="#e6edf3")

plt.tight_layout()
plt.savefig(OUT / "4_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Salvat: 4_summary_table.png")

# ── 5. Val Loss stability ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Stabilitate Val Loss — comparatie runs", fontsize=13, fontweight="bold")

for name, d in available.items():
    color = RUNS[name]["color"]
    vl_s = ema(d["val_loss"], SMOOTH)
    ax.plot(d["epochs"], d["val_loss"], color=color, linewidth=0.7, alpha=0.18)
    ax.plot(d["epochs"], vl_s, color=color, linewidth=2.2,
            label=name, alpha=0.9)

ax.set_xlabel("Epoca", fontsize=11)
ax.set_ylabel("Val Loss", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / "5_val_loss_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Salvat: 5_val_loss_stability.png")

# ── Done ──────────────────────────────────────────────────────────────────────

print(f"\nToate graficele salvate in: {OUT}")
print("\nRezumat final:")
for name, d in available.items():
    best = max(d["top1"])
    ep   = d["epochs"][int(np.argmax(d["top1"]))]
    print(f"  {name:40s}  Best: {best:.2f}%  (ep. {ep})")
