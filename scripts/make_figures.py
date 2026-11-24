"""Regenerate the README figures from files in results/.

    python scripts/make_figures.py

fig_progression.png  DS1 grouped-CV progression (numbers transcribed from
                     results/diagnose_log.txt, select_delta_log.txt, exp_svdb_log.txt)
fig_final_ds2.png    final DS2 per-class F1, int8 student, 5 seeds (train_final_v2.json)
fig_delta_sparsity.png  dV trade-off on DS1 (select_delta.csv)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical slots 1 and 2 (light surface)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def progression() -> None:
    stages = ["spike histograms\n+ RR scalars",
              "+ RR / spike-energy\nratio features",
              "+ gain normalisation\n+ per-patient template",
              "+ SVDB as extra\ntraining data"]
    acc = [89.02, 91.18, 93.70, 94.09]
    v_f1 = [0.593, 0.671, 0.806, 0.832]
    s_f1 = [0.154, 0.230, 0.257, 0.323]
    always_n = 100 * 45833 / 50979            # DS1: share of normal beats
    y = list(range(len(stages)))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    a1.hlines(y, 87.5, acc, color=GRID, lw=2, zorder=1)
    a1.scatter(acc, y, s=90, color=BLUE, zorder=3, edgecolor="#fcfcfb", linewidth=2)
    for xv, yv in zip(acc, y):
        a1.text(xv + 0.18, yv, f"{xv:.2f}%", va="center", color=INK, fontsize=10)
    a1.axvline(always_n, color=ORANGE, lw=1.4, ls="--", zorder=2)
    a1.text(always_n + 0.12, 3.52, f"always predict 'normal' = {always_n:.1f}%", color=ORANGE,
            fontsize=8.5, ha="left", va="center")
    a1.set_yticks(y)
    a1.set_yticklabels(stages)
    a1.set_xlim(87.5, 95.6)
    a1.set_ylim(-0.6, 3.8)
    a1.set_xlabel("accuracy, DS1 record-grouped 5-fold CV (%)")
    a1.set_title("Accuracy", loc="left", color=INK, fontsize=11, fontweight="bold")
    a1.grid(axis="y", visible=False)

    for series, color, name in [(v_f1, BLUE, "V (ventricular)"), (s_f1, ORANGE, "S (supraventricular)")]:
        a2.plot(series, y, "-o", color=color, lw=2, ms=7, mec="#fcfcfb", mew=1.5, label=name)
        for xv, yv in zip(series, y):
            a2.text(xv + 0.03, yv, f"{xv:.2f}", ha="left", va="center", color=INK, fontsize=9)
    a2.set_yticks(y)
    a2.set_yticklabels([])
    a2.set_xlim(0, 1.0)
    a2.set_ylim(-0.6, 3.8)
    a2.set_xlabel("F1, same CV")
    a2.set_title("Rare-class F1", loc="left", color=INK, fontsize=11, fontweight="bold")
    a2.legend(frameon=False, loc="upper right", fontsize=9)
    a2.grid(axis="y", visible=False)
    fig.suptitle("What moved the needle (gradient-boosting probe, dV = 0.1 mV, DS1 only)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_progression.png", dpi=170)
    plt.close(fig)


def final_ds2() -> None:
    d = json.loads((ROOT / "results" / "train_final_v2.json").read_text())
    s = d["summary"]["student_int8"]
    classes = ["N", "S", "V", "F"]
    names = {"N": "N  normal", "S": "S  supraventricular", "V": "V  ventricular", "F": "F  fusion"}
    mean = [s[f"f1_{c}"][0] for c in classes]
    std = [s[f"f1_{c}"][1] for c in classes]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ypos = list(range(len(classes)))[::-1]
    ax.barh(ypos, mean, xerr=std, color=BLUE, height=0.55, ecolor=INK, capsize=3,
            error_kw={"lw": 1.2}, zorder=3)
    for m, sd, yv in zip(mean, std, ypos):
        ax.text(max(m + sd, 0) + 0.02, yv, f"{m:.3f} ± {sd:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[c] for c in classes])
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("F1 on DS2 (unseen patients), mean ± std over 5 seeds")
    ax.grid(axis="y", visible=False)
    acc = s["accuracy"]
    ax.set_title("Final result on unseen patients (DS2): int8 student, 9 KB, 89.7% idle sparsity",
                 loc="left", color=INK, fontsize=11, fontweight="bold", pad=24)
    ax.text(0, 1.04, f"accuracy {100 * acc[0]:.1f} ± {100 * acc[1]:.1f}%   (a model that always says 'normal' scores 89.0%)",
            transform=ax.transAxes, color=MUTED, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_final_ds2.png", dpi=170)
    plt.close(fig)


def delta_sparsity() -> None:
    rows = list(csv.DictReader(open(ROOT / "results" / "select_delta.csv")))
    x = [100 * float(r["ds1_idle_sparsity"]) for r in rows]
    y = [100 * float(r["cv_accuracy"]) for r in rows]
    dv = [float(r["delta_mv"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4))
    ax.plot(x, y, "-o", color=BLUE, lw=2, ms=8, mec="#fcfcfb", mew=1.5, zorder=3)
    for i, (xv, yv, d) in enumerate(zip(x, y, dv)):
        last = i == len(x) - 1
        ax.text(xv + (0.35 if last else 0), yv + (0.0 if last else 0.07), f"{d:g} mV",
                ha="left" if last else "center", va="center" if last else "bottom", color=INK, fontsize=9.5)
    ax.axvline(80, color=ORANGE, lw=1.4, ls="--")
    ax.text(80.3, min(y) - 0.02, "80% sparsity gate", color=ORANGE, fontsize=9, va="bottom")
    ax.set_xlabel("idle sparsity on DS1 (% of 360 Hz slots with no spike)")
    ax.set_ylabel("DS1 grouped-CV accuracy (%)")
    ax.set_xlim(78, 97)
    ax.set_ylim(min(y) - 0.2, max(y) + 0.4)
    ax.set_title("Sparsity is nearly free: 82% → 94% idle costs ~0.5 points of accuracy",
                 loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_delta_sparsity.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    progression()
    final_ds2()
    delta_sparsity()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
