"""README banner: an event raster from real data next to the project's key facts.

Left: MIT-BIH record 119 by default (a DS1 / training record with ventricular bigeminy) through the front-end model
(±0.1 mV level crossing, 0.05 Hz high-pass). One row per heartbeat, aligned on the annotated R peak; one pixel column per
360 Hz slot from 300 ms before to 450 ms after R. A pixel is lit only where a comparator fired: up events blue, down events
orange. Ventricular beats are marked by a short aqua tick at the row start. Everything dark is time the hardware sleeps.

    python scripts/figures/make_hero.py --data-dir data/mitdb
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.afe import run_afe  # noqa: E402
from isoline.config import AAMI_MAP, AFEConfig  # noqa: E402

BG, INK, MUTED, FAINT = "#0d1117", "#e6edf3", "#9198a1", "#30363d"
UP, DN, VB = "#338ced", "#e76624", "#00a888"          # dataviz reference slots 1-3, dark steps (validated on #0d1117)
PRE, POST, NROWS = 108, 162, 264                      # 300 ms before R, 450 ms after; rows shown


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255


def raster(data_dir, rec):
    x, r, sym = data.load_mitdb_record(data_dir, rec)
    up, dn, _ = run_afe(x, replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05), np.random.default_rng(0))
    keep = [(ri, s) for ri, s in zip(r, sym) if AAMI_MAP.get(s) in ("N", "S", "V", "F") and ri - PRE >= 0 and ri + POST < len(x)]
    keep = keep[200:200 + NROWS]                     # skip the first beats (front-end settling)
    W = PRE + POST
    img = np.tile(hexrgb(BG), (NROWS, W + 10, 1))
    nv = 0
    for i, (ri, s) in enumerate(keep):
        u = up[ri - PRE:ri + POST] > 0; d = dn[ri - PRE:ri + POST] > 0
        img[i, 10:][u] = hexrgb(UP); img[i, 10:][d] = hexrgb(DN)
        img[i, 10:][u & d] = (hexrgb(UP) + hexrgb(DN)) / 2
        if AAMI_MAP[s] == "V":
            img[i, 1:6] = hexrgb(VB); nv += 1
    lit = sum(((up[ri - PRE:ri + POST] > 0) | (dn[ri - PRE:ri + POST] > 0)).sum() for ri, _ in keep)
    idle = 100 * (1 - lit / (len(keep) * W))
    return img, nv, idle


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--record", default="119"); a = ap.parse_args()
    img, nv, idle = raster(a.data_dir, a.record)
    fig = plt.figure(figsize=(13.2, 6.6), facecolor=BG)
    ax = fig.add_axes([0, 0, 0.5, 1]); ax.axis("off")
    ax.imshow(img, aspect="auto", interpolation="nearest")
    t = lambda x, y, s, **k: fig.text(x, y, s, **k)
    t(0.53, 0.80, "ISOLINE", fontfamily="Inter Display", fontweight="bold", fontsize=46, color=INK, va="top")
    t(0.532, 0.655, "event-driven arrhythmia detection", fontfamily="Inter", fontsize=17, color=MUTED, va="top")
    for k, line in enumerate(["95.6 % on unseen patients  ·  MIT-BIH inter-patient split",
                              "5.2 KB classifier  ·  sub-milliwatt analog front end",
                              f"{NROWS} heartbeats of MIT-BIH record {a.record}, one per row, aligned on R",
                              f"lit pixels are comparator events; {idle:.0f} % of this window is dark, idle time"]):
        t(0.532, 0.50 - 0.072 * k, line, fontfamily="Inter", fontsize=12.2, color=INK if k < 2 else MUTED, va="top")
    lx, ly = 0.532, 0.18
    for k, (c, lab) in enumerate(((UP, "up event"), (DN, "down event"), (VB, "ventricular beat"))):
        x0 = lx + [0, 0.105, 0.225][k]
        fig.add_artist(Line2D([x0 + 0.006], [ly], ls="", marker="o", ms=9, color=c, transform=fig.transFigure))
        t(x0 + 0.018, ly, lab, fontfamily="Inter", fontsize=11.5, color=MUTED, va="center")
    out = ROOT / "docs" / "img" / "hero.png"
    fig.savefig(out, dpi=200, facecolor=BG)                 # 2640 x 1320, crisp on high-DPI screens
    plt.close(fig)
    print(out, f"(ventricular rows: {nv}, idle {idle:.1f} %)")


if __name__ == "__main__":
    main()
