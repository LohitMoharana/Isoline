"""Dev set for non-premature S beats (pre-RR >= 0.9 x dominant RR). DS2 never loaded.
E1: train DS1 + INCART. E2: + SVDB in training (SVDB evaluated by 5 record-grouped folds, never on itself).
Measured on: SVDB held-out, LTDB (external), DS1 5-fold. Hypothesis is accepted only if non-premature S sensitivity
improves on at least two sources without an accuracy loss on any."""
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
from scripts.dev.exp_pwave import fitp  # noqa: E402
from scripts.final.final_gbt import X_of  # noqa: E402
from scripts.dev.find_nonpremature_s import nonprem_mask  # noqa: E402

D, DATA, C = sys.argv[1:4]
cfg = replace(AFEConfig(), delta_mv=0.1)
ds1, _ = data.get_splits(D)
L = lambda db, lead: build_dataset(sorted(p.stem for p in Path(DATA, db).glob("*.hea")), str(Path(DATA) / db), cfg, cache_dir=C, verbose=False, lead=lead)
d, e, v, lt = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False), L("incart", "II"), L("svdb", "ECG1"), L("ltdb", "ECG1")
yi = e["y"].astype(int); keep = np.ones(len(yi), bool)
keep[np.random.default_rng(0).permutation(np.flatnonzero(yi == 0))[40000:]] = False
yv = v["y"].astype(int); keepv = np.ones(len(yv), bool)
keepv[np.random.default_rng(1).permutation(np.flatnonzero(yv == 0))[40000:]] = False
Xd, Xe, Xv, Xl = X_of(d), X_of(e)[keep], X_of(v), X_of(lt)
y, g, yl = d["y"].astype(int), d["rec"], lt["y"].astype(int)
masks = {"DS1": nonprem_mask(d), "SVDB": nonprem_mask(v), "LTDB": nonprem_mask(lt)}


def show(name, src, yt, p):
    r = report(yt, p); m = masks[src]
    se_np = ((p == 1) & m).sum() / max(m.sum(), 1)
    return f"{src} acc {100 * r['accuracy']:.2f} S F1 {r['S']['f1']:.3f} nonprem-S Se {se_np:.3f}"


for exp in ("E1 DS1+INCART", "E2 +SVDB"):
    use_sv = exp.startswith("E2")
    # DS1 5-fold
    pd_ = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(Xd, y, g):
        Xt = [Xd[tr], Xe] + ([Xv[keepv]] if use_sv else []); yt = [y[tr], yi[keep]] + ([yv[keepv]] if use_sv else [])
        pd_[te] = fitp(np.concatenate(Xt), np.concatenate(yt)).predict(Xd[te])
    # SVDB held-out folds
    pv = np.zeros_like(yv)
    for tr, te in GroupKFold(5).split(Xv, yv, v["rec"]):
        Xt = [Xd, Xe] + ([Xv[tr][keepv[tr]]] if use_sv else []); yt = [y, yi[keep]] + ([yv[tr][keepv[tr]]] if use_sv else [])
        pv[te] = fitp(np.concatenate(Xt), np.concatenate(yt)).predict(Xv[te])
    # LTDB external
    Xt = [Xd, Xe] + ([Xv[keepv]] if use_sv else []); yt = [y, yi[keep]] + ([yv[keepv]] if use_sv else [])
    pl = fitp(np.concatenate(Xt), np.concatenate(yt)).predict(Xl)
    print(f"{exp:14s} | {show(exp, 'DS1', y, pd_)} | {show(exp, 'SVDB', yv, pv)} | {show(exp, 'LTDB', yl, pl)}", flush=True)
