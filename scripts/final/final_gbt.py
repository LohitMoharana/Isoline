"""Frozen-config evaluation of the gradient-boosted-tree classifier.

Configuration (chosen on DS1 grouped CV only, see results/exp_*_log.txt):
  features : spike histograms + template residual + 13 base causal scalars
             + long-memory RR (cols 21-25) + online per-patient morphology clusters (cols 32-37)
  data     : MIT-BIH DS1 + INCART lead II (normal beats subsampled to 40k), training only
  model    : HistGradientBoosting, 150 rounds, 31 leaves, lr 0.1, l2 1.0, sqrt-balanced weights

Without --final only the DS1 grouped-CV estimate is computed. With --final the model is
trained on all of DS1 (+INCART) and evaluated ONCE on DS2 for each of 5 INCART subsample seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402

COLS = list(range(13)) + [21, 22, 23, 24, 25] + [32, 33, 34, 35, 36, 37]


def X_of(d):
    n = len(d["y"])
    return np.concatenate([d["hist"].reshape(n, -1), d["res"].reshape(n, -1),
                           d["scal"][:, COLS]], 1).astype(np.float32)


import os
TINY = os.environ.get("TINY") == "1"          # frozen 5.2 KB config, chosen by DS1 LOPO (results/dev/exp_tiny_gbt_log.txt)


def make_clf(seed):
    if TINY:
        return HistGradientBoostingClassifier(max_iter=30, learning_rate=0.2, max_leaf_nodes=8,
                                              l2_regularization=1.0, random_state=seed, early_stopping=False)
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)


def footprint_bytes(clf) -> dict:
    internal = leaves = 0
    for it in clf._predictors:
        for tp in it:
            nd = tp.nodes
            leaves += int(nd["is_leaf"].sum())
            internal += int((~nd["is_leaf"].astype(bool)).sum())
    # internal node: feature id (1 B) + 16-bit integer threshold + 16-bit right-child offset; leaf: int8 value
    return dict(trees=sum(len(it) for it in clf._predictors), internal_nodes=internal, leaves=leaves,
                bytes=internal * 5 + leaves)


def fit(X, y, Xe, ye, seed):
    rs = np.random.default_rng(seed)
    nidx = np.flatnonzero(ye == 0)
    keep = np.ones(len(ye), bool)
    keep[rs.permutation(nidx)[40000:]] = False
    Xt, yt = np.concatenate([X, Xe[keep]]), np.concatenate([y, ye[keep]])
    return make_clf(seed).fit(Xt, yt, sample_weight=np.sqrt(compute_sample_weight("balanced", yt)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--incart-dir", default=str(ROOT / "data" / "incart"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, ds2 = data.get_splits(a.data_dir)
    d1 = build_dataset(ds1, a.data_dir, cfg, cache_dir=a.cache, verbose=False)
    recs = sorted(p.stem for p in Path(a.incart_dir).glob("*.hea"))
    e = build_dataset(recs, a.incart_dir, cfg, cache_dir=a.cache, verbose=False, lead="II")
    X, y, g = X_of(d1), d1["y"].astype(int), d1["rec"]
    Xe, ye = X_of(e), e["y"].astype(int)
    out = dict(config=dict(cols=COLS, delta_mv=0.1, incart_n_keep=40000, iters=150, leaves=31, lr=0.1))

    if not a.final:
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, g):
            pred[te] = fit(X[tr], y[tr], Xe, ye, a.seeds[0]).predict(X[te])
        r = report(y, pred)
        print(format_report(r, "== DS1 grouped CV"))
        out["ds1_cv"] = r
        (Path(a.out) / "final_gbt_dev.json").write_text(json.dumps(out, indent=2))
        return

    d2 = build_dataset(ds2, a.data_dir, cfg, cache_dir=a.cache, verbose=False)
    X2, y2 = X_of(d2), d2["y"].astype(int)
    print(f"DS2 beats {len(y2)}  always-N accuracy {np.mean(y2 == 0):.4f}  idle sparsity {d2['stats']['idle_sparsity']:.3f}")
    runs = []
    for s in a.seeds:
        clf = fit(X, y, Xe, ye, s)
        r = report(y2, clf.predict(X2))
        r["footprint"] = footprint_bytes(clf)
        runs.append(r)
        print(format_report(r, f"== DS2 seed {s}  ({r['footprint']['bytes']} B est.)"), flush=True)
    keys = ["accuracy", "macro_f1"]
    summ = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}
    for c in "NSVF":
        for m in ("f1", "se", "ppv"):
            v = [r[c][m] for r in runs]
            summ[f"{m}_{c}"] = (float(np.mean(v)), float(np.std(v)))
    print("\n== summary over seeds", a.seeds)
    for k, v in summ.items():
        print(f"  {k:9s} {v[0]:.4f} +/- {v[1]:.4f}")
    out.update(summary=summ, runs=runs, ds2_idle_sparsity=d2["stats"]["idle_sparsity"])
    (Path(a.out) / ("final_tiny_gbt.json" if TINY else "final_gbt.json")).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
