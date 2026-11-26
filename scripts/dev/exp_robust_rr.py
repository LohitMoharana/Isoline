"""Robust rhythm features for the beat classifier: the AF detector's RR statistics (previous 16 beats: irregularity,
dominance, short/long fractions, compensatory pairs, current RR / median) plus pre-RR and post-RR relative to the
dominant RR (75th percentile). Dev only: DS1 5-fold and external SVDB (trained on DS1 + INCART)."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
from scripts.final.af_detector import rr_features  # noqa: E402
from scripts.dev.exp_pwave import fitp  # noqa: E402
from scripts.final.final_gbt import X_of  # noqa: E402


def robust(d, delayed):
    r, g = d["r"], d["rec"]
    out = np.zeros((len(r), 12 if delayed else 11), np.float32)
    for rec in np.unique(g):
        k = np.flatnonzero(g == rec); t = r[k] / 360
        f = np.nan_to_num(rr_features(t), nan=0.0)
        rr = np.diff(t, prepend=t[0])
        p75 = np.array([np.percentile(rr[max(i - 16, 1):i], 75) if i > 8 else rr[i] for i in range(len(t))])
        cols = [f, (rr / np.maximum(p75, 1e-3))[:, None]]
        if delayed:
            nxt = np.append(rr[1:], rr[-1]); cols.append((nxt / np.maximum(p75, 1e-3))[:, None])
        out[k] = np.concatenate(cols, 1)
    return out


def main():
    D, INC, SV, C = sys.argv[1:5]
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, _ = data.get_splits(D)
    d = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
    v = build_dataset(sorted(p.stem for p in Path(SV).glob("*.hea")), SV, cfg, cache_dir=C, verbose=False, lead="ECG1")
    yi = e["y"].astype(int); keep = np.ones(len(yi), bool)
    keep[np.random.default_rng(0).permutation(np.flatnonzero(yi == 0))[40000:]] = False
    y, g, yv = d["y"].astype(int), d["rec"], v["y"].astype(int)
    for name, extra in (("baseline", None), ("+ robust RR (causal)", False), ("+ robust RR + next RR / dominant", True)):
        Xd, Xe, Xv = X_of(d), X_of(e), X_of(v)
        if extra is not None:
            Xd = np.concatenate([Xd, robust(d, extra)], 1); Xe = np.concatenate([Xe, robust(e, extra)], 1); Xv = np.concatenate([Xv, robust(v, extra)], 1)
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(Xd, y, g):
            pred[te] = fitp(np.concatenate([Xd[tr], Xe[keep]]), np.concatenate([y[tr], yi[keep]])).predict(Xd[te])
        a = report(y, pred)
        b = report(yv, fitp(np.concatenate([Xd, Xe[keep]]), np.concatenate([y, yi[keep]])).predict(Xv))
        print(f"{name:36s} DS1 5-fold acc {100 * a['accuracy']:.2f} S F1 {a['S']['f1']:.3f} | SVDB acc {100 * b['accuracy']:.2f} "
              f"S Se {b['S']['se']:.3f} +P {b['S']['ppv']:.3f} F1 {b['S']['f1']:.3f} V F1 {b['V']['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
