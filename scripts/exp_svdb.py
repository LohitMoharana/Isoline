"""Does extra TRAINING data (MIT-BIH Supraventricular DB) help? DS1 grouped CV only.

SVDB records are added to every training fold and never used for testing; DS2
is not loaded. Compare against the same CV without SVDB.
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
    h = np.concatenate([ds["hist"].reshape(len(ds["y"]), -1),
                        ds["res"].reshape(len(ds["y"]), -1)], 1)
    return np.concatenate([h, ds["scal"][:, :ncol]], 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mit-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--svdb-dir", default=str(ROOT / "data" / "svdb"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--svdb-lead", default="ECG1")
    a = ap.parse_args()
    cfg = replace(AFEConfig(), delta_mv=a.delta)
    ds1, _ = data.get_splits(a.mit_dir)               # DS2 ignored
    d = build_dataset(ds1, a.mit_dir, cfg, cache_dir=a.cache, verbose=False)
    recs = sorted(p.stem for p in Path(a.svdb_dir).glob("*.hea"))
    sv = build_dataset(recs, a.svdb_dir, cfg, cache_dir=a.cache, verbose=False, lead=a.svdb_lead)
    print(f"DS1 beats {len(d['y'])}  SVDB beats {len(sv['y'])}  "
          f"SVDB class counts {np.bincount(sv['y'].astype(int), minlength=4).tolist()}  "
          f"SVDB idle sparsity {sv['stats']['idle_sparsity']:.3f}", flush=True)
    X, y, g = X_of(d), d["y"].astype(int), d["rec"]
    Xs, ys = X_of(sv), sv["y"].astype(int)
    for name, use_sv in [("DS1 only", False), ("DS1 + SVDB (train only)", True)]:
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, g):
            Xtr, ytr = (np.concatenate([X[tr], Xs]), np.concatenate([y[tr], ys])) if use_sv else (X[tr], y[tr])
            w = np.sqrt(compute_sample_weight("balanced", ytr))
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                                 max_leaf_nodes=31, l2_regularization=1.0,
                                                 random_state=0).fit(Xtr, ytr, sample_weight=w)
            pred[te] = clf.predict(X[te])
        print(format_report(report(y, pred), f"== DS1 grouped-CV: {name}"), flush=True)


if __name__ == "__main__":
    main()
