"""Size-capped boosted trees (<= ~9 KB) on the round-4 features. DS1 leave-one-patient-out only; DS2 never loaded.

Footprint per model: internal node = feature id (1 B) + 16-bit integer threshold + 16-bit child offset, leaf = int8.
"""
from __future__ import annotations

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
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
from scripts.final.final_gbt import X_of, footprint_bytes  # noqa: E402

CONFIGS = [(30, 8), (50, 8), (25, 16), (60, 6), (40, 8)]


def main():
    D = sys.argv[1]; INC = sys.argv[2]; C = sys.argv[3]
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, _ = data.get_splits(D)
    d = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
    X, y, g = X_of(d), d["y"].astype(int), d["rec"]
    ye = e["y"].astype(int); keep = np.ones(len(ye), bool)
    keep[np.random.default_rng(0).permutation(np.flatnonzero(ye == 0))[40000:]] = False
    Xe, ye = X_of(e)[keep], ye[keep]
    for it, lf in CONFIGS:
        mk = lambda: HistGradientBoostingClassifier(max_iter=it, learning_rate=0.2, max_leaf_nodes=lf,
                                                    l2_regularization=1.0, random_state=0, early_stopping=False)
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(len(np.unique(g))).split(X, y, g):
            Xt = np.concatenate([X[tr], Xe]); yt = np.concatenate([y[tr], ye])
            pred[te] = mk().fit(Xt, yt, sample_weight=np.sqrt(compute_sample_weight("balanced", yt))).predict(X[te])
        Xt = np.concatenate([X, Xe]); yt = np.concatenate([y, ye])
        fp = footprint_bytes(mk().fit(Xt, yt, sample_weight=np.sqrt(compute_sample_weight("balanced", yt))))
        r = report(y, pred)
        print(f"iters {it:3d} leaves {lf:2d}  {fp['bytes']:6d} B  DS1 LOPO acc {100 * r['accuracy']:.2f}  "
              f"macroF1 {r['macro_f1']:.3f}  S {r['S']['f1']:.3f}  V {r['V']['f1']:.3f}  F {r['F']['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
