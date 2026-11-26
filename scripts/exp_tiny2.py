"""Tiny (<= 9 KB) model variants scored on two dev sets, DS2 never loaded:
DS1 leave-one-patient-out, and external SVDB (model trained on all of DS1 [+ INCART]).
Selection rule (fixed in advance): highest mean of the two accuracies among variants <= 9 KB.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
from scripts.final_gbt import footprint_bytes  # noqa: E402

R4 = list(range(13)) + [21, 22, 23, 24, 25] + [32, 33, 34, 35, 36, 37]
PORT = [47, 48, 39, 40, 41, 43, 44, 45]
VARIANTS = [  # name, hp, cols, use_incart, delayed(next-RR col 15)
    ("R4 hp0.5 +INC (round 7)", 0.5, R4, True, False),
    ("R4 hp0.5 MIT-only", 0.5, R4, False, False),
    ("R4+port hp0.5 MIT-only", 0.5, R4 + PORT, False, False),
    ("R4+port hp0.05 MIT-only", 0.05, R4 + PORT, False, False),
    ("R4+port hp0.05 +INC", 0.05, R4 + PORT, True, False),
    ("R4+port+nextRR hp0.05 MIT-only", 0.05, R4 + PORT + [15], False, True),
]


def X_of(d, cols):
    n = len(d["y"])
    return np.concatenate([d["hist"].reshape(n, -1), d["res"].reshape(n, -1), d["scal"][:, cols]], 1).astype(np.float32)


def mk():
    return HistGradientBoostingClassifier(max_iter=30, learning_rate=0.2, max_leaf_nodes=8, l2_regularization=1.0,
                                          random_state=0, early_stopping=False)


def fit(X, y):
    return mk().fit(X, y, sample_weight=np.sqrt(compute_sample_weight("balanced", y)))


def main():
    D, INC, SV, C = sys.argv[1:5]
    ds1, _ = data.get_splits(D)
    cache = {}
    def get(hp):
        if hp not in cache:
            cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=hp)
            d = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
            e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
            v = build_dataset(sorted(p.stem for p in Path(SV).glob("*.hea")), SV, cfg, cache_dir=C, verbose=False, lead="ECG1")
            ye = e["y"].astype(int); keep = np.ones(len(ye), bool)
            keep[np.random.default_rng(0).permutation(np.flatnonzero(ye == 0))[40000:]] = False
            cache[hp] = (d, e, keep, v)
        return cache[hp]
    for name, hp, cols, inc, _ in VARIANTS:
        d, e, keep, v = get(hp)
        X, y, g = X_of(d, cols), d["y"].astype(int), d["rec"]
        Xe, ye = X_of(e, cols)[keep], e["y"].astype(int)[keep]
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(len(np.unique(g))).split(X, y, g):
            Xt = np.concatenate([X[tr], Xe]) if inc else X[tr]; yt = np.concatenate([y[tr], ye]) if inc else y[tr]
            pred[te] = fit(Xt, yt).predict(X[te])
        r1 = report(y, pred)
        m = fit(np.concatenate([X, Xe]) if inc else X, np.concatenate([y, ye]) if inc else y)
        r2 = report(v["y"].astype(int), m.predict(X_of(v, cols)))
        print(f"{name:34s} {footprint_bytes(m)['bytes']:5d} B | DS1 LOPO {100 * r1['accuracy']:.2f} (S {r1['S']['f1']:.2f} V {r1['V']['f1']:.2f}) "
              f"| SVDB {100 * r2['accuracy']:.2f} (S {r2['S']['f1']:.2f} V {r2['V']['f1']:.2f}) | mean {50 * (r1['accuracy'] + r2['accuracy']):.2f}", flush=True)


if __name__ == "__main__":
    main()
