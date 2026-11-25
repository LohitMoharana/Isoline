"""Extra TRAINING databases (SVDB, INCART) vs DS1-only. DS1 grouped CV; DS2 never loaded.

Extra records are added to every training fold and never evaluated.
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
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402


def X_of(ds, ncol=13):
    n = len(ds["y"])
    return np.concatenate([ds["hist"].reshape(n, -1), ds["res"].reshape(n, -1),
                           ds["scal"][:, :ncol]], 1).astype(np.float32)


def load_extra(path, lead, cfg, cache):
    recs = sorted(p.stem for p in Path(path).glob("*.hea"))
    d = build_dataset(recs, path, cfg, cache_dir=cache, verbose=False, lead=lead)
    print(f"{Path(path).name}: {len(d['y'])} beats, classes {np.bincount(d['y'].astype(int), minlength=4).tolist()}, "
          f"idle sparsity {d['stats']['idle_sparsity']:.3f}", flush=True)
    return X_of(d), d["y"].astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mit-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--svdb-dir", default=str(ROOT / "data" / "svdb"))
    ap.add_argument("--incart-dir", default=str(ROOT / "data" / "incart"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--extra-weight", type=float, default=1.0, help="sample-weight multiplier for extra beats")
    a = ap.parse_args()
    cfg = replace(AFEConfig(), delta_mv=a.delta)
    ds1, _ = data.get_splits(a.mit_dir)
    d = build_dataset(ds1, a.mit_dir, cfg, cache_dir=a.cache, verbose=False)
    X, y, g = X_of(d), d["y"].astype(int), d["rec"]
    ex = {"SVDB": load_extra(a.svdb_dir, "ECG1", cfg, a.cache),
          "INCART": load_extra(a.incart_dir, "II", cfg, a.cache)}
    variants = [("DS1 only", []), ("DS1 + INCART", ["INCART"]), ("DS1 + SVDB + INCART", ["SVDB", "INCART"])]
    for name, extras in variants:
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, g):
            Xs = [X[tr]] + [ex[k][0] for k in extras]
            ys = [y[tr]] + [ex[k][1] for k in extras]
            isx = [np.ones(len(tr))] + [np.full(len(ex[k][1]), a.extra_weight) for k in extras]
            Xtr, ytr, mult = np.concatenate(Xs), np.concatenate(ys), np.concatenate(isx)
            w = np.sqrt(compute_sample_weight("balanced", ytr)) * mult
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                                 l2_regularization=1.0, random_state=0).fit(Xtr, ytr, sample_weight=w)
            pred[te] = clf.predict(X[te])
        print(format_report(report(y, pred), f"== DS1 grouped-CV: {name} (extra weight {a.extra_weight})"), flush=True)


if __name__ == "__main__":
    main()
