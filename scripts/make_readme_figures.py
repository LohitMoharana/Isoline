"""README visuals, rendered twice: for GitHub's light theme and for its dark theme (served with <picture> and
prefers-color-scheme). Every number is read from results/*.json or transcribed from logs in results/ (cited inline).

    python scripts/make_readme_figures.py --data-dir data/mitdb

Palette: the dataviz reference categorical slots 1-3 (blue, orange, aqua), validated with validate_palette.js against
#ffffff (light) and #0d1117 (GitHub dark): all-pairs CVD dE >= 12.6 / 12.8, normal-vision >= 23.0 / 19.4. In light
mode orange and aqua sit below 3:1 on white, so every series is also labelled directly.
Neutral ink is used for references and for "not ours"; colour always means a series.
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
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.afe import run_afe  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
R = lambda f: json.loads((ROOT / "results" / f).read_text())

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", muted="#59636e", faint="#d8dee4", band="#f3f5f7", neutral="#8c959f",
                  s1="#257de0", s2="#f47230", s3="#00b894"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9198a1", faint="#262c36", band="#151b23", neutral="#6e7681",
                 s1="#338ced", s2="#e76624", s3="#00a888"),
}
SANS, DISPLAY, SERIF = "Inter", "Inter Display", "Crimson Text"


def style(t):
    plt.rcParams.update({
        "font.family": SANS, "font.size": 9.5, "text.color": t["ink"],
        "axes.edgecolor": t["faint"], "axes.labelcolor": t["muted"], "axes.labelsize": 9,
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": t["faint"], "grid.linewidth": 0.7,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": t["ink"],
    })


def heading(fig, t, title, sub=None, x=0.012):
    fig.text(x, 0.975, title, fontfamily=DISPLAY, fontsize=13, fontweight="semibold", color=t["ink"], va="top")
    if sub:
        fig.text(x, 0.905, sub, fontsize=9, color=t["muted"], va="top")


def save(fig, name, theme):
    fig.savefig(OUT / f"{name}_{theme}.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- data helpers
def ecg_segment(data_dir, rec="101", t0=10.0, dur=6.0):
    x, r, sym = data.load_mitdb_record(data_dir, rec)
    cfg = replace(AFEConfig(), delta_mv=0.1)
    up, dn, filt = run_afe(x, cfg, np.random.default_rng(0))
    s, e = int(t0 * 360), int((t0 + dur) * 360)
    t = np.arange(e - s) / 360
    return t, filt[s:e], up[s:e].astype(bool), dn[s:e].astype(bool)


# ----------------------------------------------------------------------------------------------- figures
def hero(seg, t, theme):
    tt, y, up, dn = seg
    fig = plt.figure(figsize=(12, 3.3))
    fig.text(0.02, 0.80, "Isoline", fontfamily=SERIF, fontsize=50, color=t["ink"], va="top")
    fig.text(0.022, 0.47, "Event-driven ECG arrhythmia detection", fontfamily=DISPLAY, fontsize=13.5,
             fontweight="medium", color=t["ink"], va="top")
    fig.text(0.022, 0.37, "It computes only when the heart moves.", fontsize=11, fontweight="light", color=t["muted"], va="top")
    ax = fig.add_axes([0.33, 0.16, 0.66, 0.78]); ax.axis("off")
    active = np.convolve((up | dn).astype(float), np.ones(40), "same") > 0
    ax.fill_between(tt, -1.0, 2.3, where=~active, color=t["band"], step="mid", lw=0)
    ax.plot(tt, y, color=t["ink"], lw=1.1, alpha=0.92, solid_capstyle="round")
    ax.vlines(tt[up], 1.85, 2.2, color=t["s1"], lw=1.0)
    ax.vlines(tt[dn], -0.92, -0.57, color=t["s2"], lw=1.0)
    ax.set_xlim(tt[0], tt[-1]); ax.set_ylim(-1.0, 2.3)
    idle = 100 * (1 - (up | dn).mean())
    lx = 0.335
    fig.add_artist(Line2D([lx, lx + 0.012], [0.075, 0.075], color=t["s1"], lw=2, transform=fig.transFigure))
    fig.text(lx + 0.017, 0.075, "up event", fontsize=8.5, color=t["muted"], va="center")
    fig.add_artist(Line2D([lx + 0.085, lx + 0.097], [0.075, 0.075], color=t["s2"], lw=2, transform=fig.transFigure))
    fig.text(lx + 0.102, 0.075, "down event", fontsize=8.5, color=t["muted"], va="center")
    fig.text(0.99, 0.075, f"shaded: idle  ·  {idle:.0f} % of 360 Hz slots carry no event  ·  MIT-BIH record 101",
             fontsize=8.5, color=t["muted"], va="center", ha="right")
    save(fig, "hero", theme)


def animation_gif(seg, t, theme):
    tt, y, up, dn = seg
    win = int(3 * 360)
    fig = plt.figure(figsize=(8, 2.5), facecolor=t["bg"])
    ax = fig.add_axes([0.01, 0.2, 0.98, 0.78], facecolor=t["bg"]); ax.axis("off")
    line, = ax.plot([], [], color=t["ink"], lw=1.3)
    ups = ax.vlines([], 0, 0, color=t["s1"], lw=1.2)
    dns = ax.vlines([], 0, 0, color=t["s2"], lw=1.2)
    dot = fig.text(0.02, 0.07, "●", fontsize=10, color=t["neutral"], va="center")
    txt = fig.text(0.045, 0.07, "", fontsize=9.5, color=t["muted"], va="center")
    steps = list(range(win, len(tt), 18))

    def frame(k):
        e = steps[k]; s = e - win
        line.set_data(tt[s:e], y[s:e]); ax.set_xlim(tt[s], tt[e - 1]); ax.set_ylim(-1.1, 2.3)
        uu, dd = np.flatnonzero(up[s:e]) + s, np.flatnonzero(dn[s:e]) + s
        ups.set_segments([[(tt[i], 1.85), (tt[i], 2.2)] for i in uu])
        dns.set_segments([[(tt[i], -1.0), (tt[i], -0.65)] for i in dd])
        busy = (up[e - 36:e] | dn[e - 36:e]).any()
        dot.set_color(t["s3"] if busy else t["neutral"])
        txt.set_text("computing: QRS burst → integer features → 5 KB trees" if busy else "idle: no events, the processor sleeps")
        txt.set_color(t["ink"] if busy else t["muted"])
        return line, ups, dns, txt, dot

    ani = animation.FuncAnimation(fig, frame, frames=len(steps), blit=False)
    ani.save(OUT / f"isoline_live_{theme}.gif", writer=animation.PillowWriter(fps=20), dpi=90,
             savefig_kwargs={"facecolor": t["bg"], "transparent": False})
    plt.close(fig)


def size_vs_accuracy(t, theme):
    best = R("final_af_detector.json")
    pts = [  # (label, KB, acc, ours)
        ("Isoline int8 CNN", 9.1, 93.7, True),
        ("Isoline tiny + AF", 10.3, 100 * best["tiny_5KB"]["summary"]["accuracy"][0], True),
        ("Isoline full + AF", 55.0, 100 * best["round4"]["summary"]["accuracy"][0], True),
        ("LUT network (2026), ~16 KB est.", 16, 94.41, False),
        ("RF paper, our clean re-run", 304, 95.39, False),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.5)); fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.13)
    heading(fig, t, "Accuracy against model size", "Isoline reads sparse comparator events; the other methods read the full-rate ECG.")
    for yv, lab, ls in ((96.15, "best reported on this split, 96.15 %  (size not reported)", (0, (4, 3))),
                        (89.04, "always predict ‘normal’, 89.0 %", (0, (1, 2)))):
        ax.axhline(yv, color=t["neutral"], lw=1, ls=ls, zorder=1)
        ax.text(5.3, yv + 0.12, lab, color=t["muted"], fontsize=8.3)
    for lab, kb, acc, ours in pts:
        ax.scatter(kb, acc, s=110 if ours else 70, color=t["s1"] if ours else t["neutral"], marker="o" if ours else "s",
                   edgecolor=t["bg"], linewidth=2, zorder=3)
        if lab == "Isoline int8 CNN":                       # label below the point, clear of the LUT label
            ax.text(kb, acc - 0.2, lab, va="top", ha="center", fontsize=8.6, color=t["ink"], fontweight="medium")
            ax.text(kb, acc - 0.55, f"{acc:.2f} %", va="top", ha="center", fontsize=8.3, color=t["muted"])
            continue
        dy = {"Isoline tiny + AF": 0.45}.get(lab, 0)
        ax.text(kb * 1.13, acc + dy, lab, va="bottom", fontsize=8.6, color=t["ink"] if ours else t["muted"],
                fontweight="medium" if ours else "normal")
        ax.text(kb * 1.13, acc + dy, f"{acc:.2f} %", va="top", fontsize=8.3, color=t["muted"])
    ax.set_xscale("log"); ax.set_xlim(5, 2000); ax.set_ylim(88.5, 96.8)
    ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(["10 KB", "30 KB", "100 KB", "300 KB", "1 MB"])
    ax.set_ylabel("accuracy on unseen patients (DS2, %)")
    save(fig, "size_vs_accuracy", theme)


def per_class(t, theme):
    b = R("final_af_detector.json")["round4"]["summary"]
    rep = R("replicate_rf.json")["ds2_clean"]
    rf = {c: np.mean([r[c]["f1"] for r in rep]) for c in "NSVF"}
    cls = "NSVF"; names = ["N  normal", "S  supraventricular", "V  ventricular", "F  fusion"]
    ours = [b[f"f1_{c}"][0] for c in cls]; theirs = [rf[c] for c in cls]
    x = np.arange(4); w = 0.34
    fig, ax = plt.subplots(figsize=(8.6, 4.0)); fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.12)
    heading(fig, t, "Per class: stronger on V, weaker on S", "F1 on DS2 (22 unseen patients)")
    ax.bar(x - w / 2 - 0.015, ours, w, color=t["s1"], zorder=3)
    ax.bar(x + w / 2 + 0.015, theirs, w, color=t["neutral"], zorder=3)
    for i in range(4):
        ax.text(x[i] - w / 2 - 0.015, ours[i] + 0.025, f"{ours[i]:.2f}", ha="center", fontsize=8.6, color=t["ink"])
        ax.text(x[i] + w / 2 + 0.015, theirs[i] + 0.025, f"{theirs[i]:.2f}", ha="center", fontsize=8.6, color=t["muted"])
    ax.set_xticks(x); ax.set_xticklabels(names, color=t["ink"]); ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for k, (c, lab) in enumerate(((t["s1"], "Isoline, comparator events (~55 KB)"), (t["neutral"], "RF paper, clean re-run on the full ECG"))):
        fig.add_artist(Line2D([0.62, 0.635], [0.93 - 0.06 * k] * 2, color=c, lw=6, transform=fig.transFigure, solid_capstyle="butt"))
        fig.text(0.643, 0.93 - 0.06 * k, lab, fontsize=8.6, color=t["ink"], va="center")
    save(fig, "per_class", theme)


def journey(t, theme):
    # DS2 accuracy of every frozen annotated-beat evaluation, in order (results/*.json; docs/EXPERIMENTS.md)
    steps = [("R1 int8 CNN", 93.7, "kept"), ("R2 + INCART CNN", 92.3, "rejected"), ("R4 trees + patient clusters", 94.76, "kept"),
             ("R5 two-stage", 93.0, "rejected"), ("R6 RF paper re-run", 95.39, "baseline"), ("R7 5 KB trees", 93.27, "kept"),
             ("R8 5 KB, MIT only", 92.65, "rejected"), ("R9 + rhythm flag", 95.49, "kept"), ("R10 + AF detector", 95.64, "kept"),
             ("R11 + robust RR", 95.25, "rejected"), ("R12 S-sensitive mode", 94.18, "option")]
    fig, ax = plt.subplots(figsize=(10.5, 4.6)); fig.subplots_adjust(left=0.065, right=0.985, top=0.8, bottom=0.27)
    heading(fig, t, "Every frozen evaluation on unseen patients",
            "Each configuration was chosen without DS2, then frozen and evaluated once. Failures stay in the record.")
    xs = np.arange(len(steps))
    best = np.maximum.accumulate([a if k == "kept" else 0 for _, a, k in steps])
    ax.step(xs, np.where(best > 0, best, np.nan), where="post", color=t["s1"], lw=1.2, alpha=0.45, zorder=1)
    ax.axhline(96.15, color=t["neutral"], lw=1, ls=(0, (4, 3)))
    ax.text(len(steps) - 0.6, 96.24, "best reported, 96.15 %", ha="right", color=t["muted"], fontsize=8.3)
    sty = {"kept": dict(color=t["s1"], marker="o"), "rejected": dict(color=t["s2"], marker="o"),
           "baseline": dict(color=t["neutral"], marker="s"), "option": dict(facecolor=t["bg"], edgecolor=t["s1"], marker="o")}
    for i, (lab, a, k) in enumerate(steps):
        kw = dict(sty[k]); kw.setdefault("edgecolor", t["bg"])
        ax.scatter(i, a, s=85, linewidth=2 if k == "option" else 1.8, zorder=3, **kw)
        ax.text(i, a + 0.24, f"{a:.2f}", ha="center", fontsize=8.2, color=t["ink"] if k == "kept" else t["muted"])
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in steps], rotation=32, ha="right", fontsize=8.3)
    ax.set_ylim(91.8, 96.7); ax.set_ylabel("DS2 accuracy (%)")
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=t["s1"], label="kept"),
               Line2D([], [], ls="", marker="o", ms=7, mfc=t["bg"], mec=t["s1"], mew=1.8, label="kept as an option"),
               Line2D([], [], ls="", marker="o", ms=7, color=t["s2"], label="rejected"),
               Line2D([], [], ls="", marker="s", ms=6.5, color=t["neutral"], label="published method, re-run")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=4, handletextpad=0.3, columnspacing=1.2)
    save(fig, "journey", theme)


def noise(t, theme):
    rows = R("noise_stress.json")
    clean = [r for r in rows if r["noise"] == "clean"][0]
    fig, ax = plt.subplots(figsize=(8.0, 4.0)); fig.subplots_adjust(left=0.09, right=0.8, top=0.78, bottom=0.14)
    heading(fig, t, "Noise stress test", "V-beat F1 of the 5.2 KB model with real MIT-BIH noise-stress recordings added")
    ax.axhline(clean["f1_V"], color=t["neutral"], ls=(0, (1, 2)), lw=1)
    ax.text(24.5, clean["f1_V"] + 0.008, f"clean  {clean['f1_V']:.2f}", fontsize=8.3, color=t["muted"])
    for kind, c, name in (("bw", t["s1"], "baseline wander"), ("ma", t["s2"], "muscle artefact"), ("em", t["s3"], "electrode motion")):
        rr = sorted([r for r in rows if r["noise"] == kind], key=lambda r: -r["snr_db"])
        xs = [r["snr_db"] for r in rr]; ys = [r["f1_V"] for r in rr]
        ax.plot(xs, ys, "-o", color=c, lw=2, ms=5.5, mec=t["bg"], mew=1.5, solid_capstyle="round")
        ax.text(xs[-1] - 0.9, ys[-1], f"{name}  {ys[-1]:.2f}", va="center", ha="left", fontsize=8.5, color=t["ink"])
    ax.set_xlim(25, -3.5); ax.set_xlabel("signal-to-noise ratio (dB)   →  noisier"); ax.set_ylabel("V F1 on DS2")
    ax.set_ylim(0.65, 1.0); ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    save(fig, "noise", theme)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb")); a = ap.parse_args()
    seg_hero = ecg_segment(a.data_dir)
    seg_anim = ecg_segment(a.data_dir, t0=20.0, dur=8.0)
    for name, t in THEMES.items():
        style(t)
        hero(seg_hero, t, name); size_vs_accuracy(t, name); per_class(t, name); journey(t, name); noise(t, name)
        animation_gif(seg_anim, t, name)
    print("wrote", sorted(p.name for p in OUT.iterdir()))
