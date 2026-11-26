"""README visuals. All numbers are read from results/*.json or transcribed from logs in results/ (cited inline).

    python scripts/make_readme_figures.py --data-dir data/mitdb
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.afe import run_afe  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a9a8a3"
DARK, DINK, DMUTED = "#0f1115", "#f2f2ef", "#9a9a94"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8})
R = lambda f: json.loads((ROOT / "results" / f).read_text())


def ecg_segment(data_dir, rec="101", t0=10.0, dur=6.0):
    x, r, sym = data.load_mitdb_record(data_dir, rec)
    cfg = replace(AFEConfig(), delta_mv=0.1)
    up, dn, filt = run_afe(x, cfg, np.random.default_rng(0))
    s, e = int(t0 * 360), int((t0 + dur) * 360)
    t = np.arange(e - s) / 360
    return t, filt[s:e], up[s:e].astype(bool), dn[s:e].astype(bool)


def hero(data_dir):
    t, y, up, dn = ecg_segment(data_dir)
    fig = plt.figure(figsize=(12, 3.6), facecolor=DARK)
    fig.text(0.015, 0.93, "Isoline", color=DINK, fontsize=26, fontweight="bold", va="top")
    fig.text(0.17, 0.885, "event-driven ECG arrhythmia detection  ·  compute only when the heart moves",
             color=DMUTED, fontsize=12, va="top")
    ax = fig.add_axes([0, 0.12, 1, 0.62], facecolor=DARK)
    active = np.convolve((up | dn).astype(float), np.ones(40), "same") > 0
    ax.fill_between(t, -1.2, 2.4, where=~active, color="#1a1d24", step="mid", lw=0)
    ax.plot(t, y, color=DINK, lw=1.4, alpha=0.9)
    ax.vlines(t[up], 1.75, 2.15, color=BLUE, lw=1.1)
    ax.vlines(t[dn], -0.95, -0.55, color=ORANGE, lw=1.1)
    idle = 100 * (1 - (up | dn).mean())
    fig.text(0.985, 0.035, f"blue / orange ticks = comparator up / down events   ·   grey bands = idle   ·   "
             f"{idle:.0f}% of 360 Hz slots carry no event (MIT-BIH record 101)", color=DMUTED, fontsize=8.5, ha="right")
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(-1.2, 2.4); ax.axis("off")
    fig.savefig(OUT / "hero.png", dpi=160, facecolor=DARK)
    plt.close(fig)


def animation_gif(data_dir):
    t, y, up, dn = ecg_segment(data_dir, t0=20.0, dur=8.0)
    win = int(3 * 360)
    fig, ax = plt.subplots(figsize=(8, 2.6), facecolor=DARK)
    ax.set_facecolor(DARK); ax.axis("off")
    line, = ax.plot([], [], color=DINK, lw=1.4)
    ups = ax.vlines([], 0, 0, color=BLUE, lw=1.2)
    dns = ax.vlines([], 0, 0, color=ORANGE, lw=1.2)
    txt = ax.text(0.01, 0.02, "", transform=ax.transAxes, color=DMUTED, fontsize=9.5, va="bottom", family="DejaVu Sans Mono")
    ax.set_ylim(-1.2, 2.4)
    steps = list(range(win, len(t), 18))

    def frame(k):
        e = steps[k]; s = e - win
        line.set_data(t[s:e], y[s:e]); ax.set_xlim(t[s], t[e - 1])
        uu, dd = np.flatnonzero(up[s:e]) + s, np.flatnonzero(dn[s:e]) + s
        ups.set_segments([[(t[i], 1.75), (t[i], 2.15)] for i in uu])
        dns.set_segments([[(t[i], -0.95), (t[i], -0.55)] for i in dd])
        recent = (up[e - 36:e] | dn[e - 36:e]).any()
        txt.set_text(("● compute: QRS burst -> features -> 5 KB trees" if recent else "○ idle: no events, compute asleep"))
        txt.set_color(GREEN if recent else DMUTED)
        ax.set_ylim(-1.7, 2.4)
        return line, ups, dns, txt

    ani = animation.FuncAnimation(fig, frame, frames=len(steps), blit=False)
    ani.save(OUT / "isoline_live.gif", writer=animation.PillowWriter(fps=20), dpi=90, savefig_kwargs={"facecolor": DARK})
    plt.close(fig)


def size_vs_accuracy():
    best = R("final_af_detector.json")
    pts = [  # (label, KB, acc, colour, ours)
        ("Isoline int8 CNN", 9.1, 93.7, BLUE, True),
        ("Isoline tiny + AF", 10.3, 100 * best["tiny_5KB"]["summary"]["accuracy"][0], BLUE, True),
        ("Isoline full + AF", 55.0, 100 * best["round4"]["summary"]["accuracy"][0], BLUE, True),
        ("LUT network (2026), ~16 KB est.", 16, 94.41, GRAY, False),
        ("RF paper, our clean re-run", 304, 95.39, GRAY, False),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.axhline(96.15, color=ORANGE, lw=1.3, ls="--")
    ax.text(5.2, 96.22, "best reported on this split: 96.15 %  (full-rate ECG, size not reported)", color=ORANGE, fontsize=8.5)
    ax.axhline(89.04, color=GRAY, lw=1, ls=":")
    ax.text(5.2, 89.15, "always predict 'normal': 89.0 %", color=MUTED, fontsize=8.5)
    for lab, kb, acc, c, ours in pts:
        ax.scatter(kb, acc, s=140 if ours else 90, color=c, edgecolor=SURF, linewidth=2, zorder=3,
                   marker="o" if ours else "s")
        dy = {"Isoline tiny + AF": 0.42, "LUT network (2026), ~16 KB est.": 0.0}.get(lab, 0)
        ax.text(kb * 1.12, acc + dy, f"{lab}\n{acc:.2f} %", va="center", fontsize=8.8, color=INK if ours else MUTED)
    ax.set_xscale("log"); ax.set_xlim(5, 2000); ax.set_ylim(88.5, 96.8)
    ax.set_xlabel("model size (KB, log scale)"); ax.set_ylabel("accuracy on unseen patients, DS2 (%)")
    ax.set_title("Accuracy vs size: Isoline (blue) runs on sparse comparator events; the others read the full ECG",
                 loc="left", color=INK, fontsize=10.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "size_vs_accuracy.png", dpi=170); plt.close(fig)


def per_class():
    b = R("final_af_detector.json")["round4"]["summary"]
    rep = R("replicate_rf.json")["ds2_clean"]
    rf = {c: np.mean([r[c]["f1"] for r in rep]) for c in "NSVF"}
    cls = ["N", "S", "V", "F"]; names = ["N normal", "S supraventricular", "V ventricular", "F fusion"]
    ours = [b[f"f1_{c}"][0] for c in cls]; theirs = [rf[c] for c in cls]
    x = np.arange(4); w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - w / 2 - 0.01, ours, w, color=BLUE, label="Isoline (events, ~55 KB)", zorder=3)
    ax.bar(x + w / 2 + 0.01, theirs, w, color=GRAY, label="RF paper, clean re-run (full ECG)", zorder=3)
    for i in range(4):
        ax.text(x[i] - w / 2, ours[i] + 0.02, f"{ours[i]:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(x[i] + w / 2, theirs[i] + 0.02, f"{theirs[i]:.2f}", ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.12); ax.set_ylabel("F1 on DS2")
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Per class: better at V, worse at S", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "per_class.png", dpi=170); plt.close(fig)


def journey():
    # DS2 accuracy of every frozen evaluation, in order (results/*.json; README "Experiment log")
    steps = [("R1 int8 CNN", 93.7, True), ("R2 +INCART CNN", 92.3, False), ("R4 trees + patient clusters", 94.76, True),
             ("R5 two-stage", 93.0, False), ("R6 RF paper re-run*", 95.39, None), ("R7 5 KB trees", 93.27, True),
             ("R8 5 KB MIT-only", 92.65, False), ("R9 + rhythm flag", 95.49, True), ("R10 + AF detector", 95.64, True),
             ("R11 + robust RR", 95.25, False)]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    xs = np.arange(len(steps))
    best = np.maximum.accumulate([a if k else 0 for _, a, k in steps])
    ax.step(xs, np.where(best > 0, best, np.nan), where="post", color=BLUE, lw=1.5, alpha=0.5, zorder=1)
    for i, (lab, a, kept) in enumerate(steps):
        c = BLUE if kept else (GRAY if kept is None else ORANGE)
        ax.scatter(i, a, s=110, color=c, edgecolor=SURF, linewidth=2, zorder=3, marker="o" if kept is not None else "s")
        ax.text(i, a + 0.22, f"{a:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in steps], rotation=28, ha="right", fontsize=8.5)
    ax.set_ylim(91.8, 96.6); ax.set_ylabel("DS2 accuracy (%)"); ax.grid(axis="x", visible=False)
    ax.axhline(96.15, color=ORANGE, lw=1, ls="--"); ax.text(len(steps) - 0.5, 96.22, "best reported 96.15 %", ha="right", color=ORANGE, fontsize=8.5)
    ax.set_title("Every frozen DS2 evaluation, kept (blue) and rejected (orange); configs were chosen without DS2",
                 loc="left", color=INK, fontsize=10.5, fontweight="bold")
    ax.text(0, 92.0, "* baseline: re-implementation of a published method", fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "journey.png", dpi=170); plt.close(fig)


def noise():
    rows = R("noise_stress.json")
    clean = [r for r in rows if r["noise"] == "clean"][0]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for kind, c, name in (("bw", GREEN, "baseline wander"), ("ma", BLUE, "muscle artefact"), ("em", ORANGE, "electrode motion")):
        rr = sorted([r for r in rows if r["noise"] == kind], key=lambda r: -r["snr_db"])
        xs = [r["snr_db"] for r in rr]; ys = [r["f1_V"] for r in rr]
        ax.plot(xs, ys, "-o", color=c, lw=2, ms=6, mec=SURF, mew=1.5, label=name)
        ax.text(xs[-1] - 0.7, ys[-1], f"{ys[-1]:.2f}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.axhline(clean["f1_V"], color=GRAY, ls=":", lw=1); ax.text(24, clean["f1_V"] + 0.01, "clean", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xlim(25, -3.5); ax.set_xlabel("signal-to-noise ratio (dB), noisier →"); ax.set_ylabel("V F1 on DS2")
    ax.set_ylim(0.65, 1.0); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Noise stress test (5.2 KB model, real NSTDB noise)", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "noise.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb")); a = ap.parse_args()
    hero(a.data_dir); size_vs_accuracy(); per_class(); journey(); noise(); animation_gif(a.data_dir)
    print("wrote", sorted(p.name for p in OUT.iterdir()))
