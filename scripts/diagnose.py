"""Diagnostics on the spike-feature baseline using DS1 only (grouped CV by record).

Compares class weighting and extra causal RR / spike-energy ratio features.
DS2 is never loaded here, so nothing found by this script is tuned on test data.
"""
from __future__ import annotations

import argparse
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
from isoline.config import CLASSES, AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402


def feats(ds, extra: bool, ncol: int = 13):
    x = np.concatenate([ds["hist"].reshape(len(ds["y"]), -1),
                        ds["res"].reshape(len(ds["y"]), -1)], 1).astype(np.float32)
    s = ds["scal"][:, :ncol].astype(np.float32)
    cols = [x, s]
    if False:
        pre, ew, dl, bs, se = (s[:, i] for i in range(5))
        cols.append(np.stack([pre * 32 / np.maximum(ew, 1),
                              bs * 32 / np.maximum(se, 1),
                              np.abs(dl)], 1))
    return np.concatenate(cols, 1)


def cv(ds, extra, weight, folds=5, max_iter=150, ncol: int = 13):
    X, y, g = feats(ds, extra, ncol), ds["y"].astype(int), ds["rec"]
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(folds).split(X, y, g):
        w = None
        if weight == "sqrt":
            w = np.sqrt(compute_sample_weight("balanced", y[tr]))
        elif weight == "balanced":
            w = compute_sample_weight("balanced", y[tr])
        clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             random_state=0)
        clf.fit(X[tr], y[tr], sample_weight=w)
        pred[te] = clf.predict(X[te])
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--delta", type=float, default=0.1)
    a = ap.parse_args()
    ds1, _ = data.get_splits(a.data_dir)
    ds = build_dataset(ds1, a.data_dir, replace(AFEConfig(), delta_mv=a.delta),
                       cache_dir=a.cache, verbose=False)
    print(f"DS1 beats {len(ds['y'])}  idle sparsity {ds['stats']['idle_sparsity']:.3f}")
    results = {}
    for name, extra, weight in [("sqrt-balanced", False, "sqrt"), ("unweighted", False, None),
                                ("balanced", False, "balanced"),
                                ("sqrt-balanced + ratio feats", True, "sqrt"),
                                ("unweighted + ratio feats", True, None)]:
        pred = cv(ds, extra, weight)
        results[name] = pred
        print(format_report(report(ds["y"], pred), f"== DS1 grouped-CV: {name}"), flush=True)

    # per-record view for the baseline: where do the errors come from?
    pred = results["sqrt-balanced"]
    print("\nper-record (sqrt-balanced): rec  n  acc  N->nonN  S_se  V_se")
    for rec in sorted(set(ds["rec"])):
        m = ds["rec"] == rec
        y, p = ds["y"][m], pred[m]
        n_err = int(np.sum((y == 0) & (p != 0)))
        s_se = np.mean(p[y == 1] == 1) if np.any(y == 1) else float("nan")
        v_se = np.mean(p[y == 2] == 2) if np.any(y == 2) else float("nan")
        print(f"  {rec} {m.sum():5d} {np.mean(y == p):.3f}  N->other {n_err:4d}  "
              f"S {s_se:.2f} (n={np.sum(y == 1)})  V {v_se:.2f} (n={np.sum(y == 2)})")


if __name__ == "__main__":
    main()
