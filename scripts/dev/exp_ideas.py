"""DS1-only grouped-CV experiments: second lead and per-class score scaling.

DS2 is never loaded. Scaling factors are tuned with nested leave-one-fold-out OOF
probabilities so the reported number is not tuned on the data it is scored on.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402


def block(ds, scal_cols):
    n = len(ds["y"])
    return np.concatenate([ds["hist"].reshape(n, -1), ds["res"].reshape(n, -1),
                           ds["scal"][:, scal_cols]], 1).astype(np.float32)


def oof_proba(X, y, g, folds=5, max_iter=150):
    P = np.zeros((len(y), 4))
    for tr, te in GroupKFold(folds).split(X, y, g):
        w = np.sqrt(compute_sample_weight("balanced", y[tr]))
        clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(X[tr], y[tr], sample_weight=w)
        P[te] = clf.predict_proba(X[te])
    return P


def macro_f1(y, p):
    return report(y, p)["macro_f1"]


def tuned_scaling(P, y, g, folds=5):
    grid = [0.5, 1, 1.5, 2, 3, 4, 6]
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(folds).split(P, y, g):
        best, bs = -1, (1, 1, 1, 1)
        for s, v, f in product(grid, grid, [1, 2, 4]):
            sc = np.array([1, s, v, f])
            m = macro_f1(y[tr], (P[tr] * sc).argmax(1))
            if m > best:
                best, bs = m, sc
        pred[te] = (P[te] * np.array(bs)).argmax(1)
    return pred


def show(name, y, p):
    r = report(y, p)
    print(f"{name:34s} acc {100 * r['accuracy']:.2f}  macroF1 {r['macro_f1']:.3f}  "
          f"S F1 {r['S']['f1']:.3f} (Se {r['S']['se']:.2f})  "
          f"V F1 {r['V']['f1']:.3f}  F F1 {r['F']['f1']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--delta", type=float, default=0.1)
    a = ap.parse_args()
    ds1, _ = data.get_splits(a.data_dir)
    cfg = replace(AFEConfig(), delta_mv=a.delta)
    d1 = build_dataset(ds1, a.data_dir, cfg, cache_dir=a.cache, verbose=False)
    y, g = d1["y"].astype(int), d1["rec"]
    X1 = block(d1, list(range(13)))
    P1 = oof_proba(X1, y, g)
    show("MLII only, argmax", y, P1.argmax(1))
    show("MLII only, nested class scaling", y, tuned_scaling(P1, y, g))
    try:
        d2 = build_dataset(ds1, a.data_dir, cfg, cache_dir=a.cache, verbose=False, lead="SECOND")
        assert (d2["y"] == d1["y"]).all() and (d2["rec"] == d1["rec"]).all()
        X2 = np.concatenate([X1, block(d2, [3, 4, 6, 9, 10, 11, 12])], 1)
        P2 = oof_proba(X2, y, g)
        show("MLII + 2nd lead, argmax", y, P2.argmax(1))
        show("MLII + 2nd lead, nested scaling", y, tuned_scaling(P2, y, g))
    except Exception as e:  # noqa: BLE001
        print("second lead failed:", repr(e))


if __name__ == "__main__":
    main()
