"""External development set: train on MIT-BIH DS1 + INCART, evaluate on SVDB (78 other patients).

Used for design choices that DS1 CV cannot measure (DS1's S beats are mostly premature). DS2 never loaded.
--fine-delta adds histograms/residuals from a second, finer level-crossing band, restricted to the
pre-R (P-wave) bins.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402

COLS = list(range(13)) + [21, 22, 23, 24, 25] + [32, 33, 34, 35, 36, 37]


def X_of(d, fine=None, pbins=12):
    n = len(d["y"])
    parts = [d["hist"].reshape(n, -1), d["res"].reshape(n, -1), d["scal"][:, COLS]]
    if fine is not None:
        parts += [fine["hist"][:, :, :pbins].reshape(n, -1), fine["res"][:, :, :pbins].reshape(n, -1)]
    return np.concatenate(parts, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mit-dir", required=True)
    ap.add_argument("--incart-dir", default="/home/claude/data/incart")
    ap.add_argument("--svdb-dir", default="/home/claude/data/svdb")
    ap.add_argument("--cache", default="/tmp/cache")
    ap.add_argument("--fine-delta", type=float, default=0.0)
    a = ap.parse_args()
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, _ = data.get_splits(a.mit_dir)
    sets = {"mit": (ds1, a.mit_dir, "MLII"),
            "inc": (sorted(p.stem for p in Path(a.incart_dir).glob("*.hea")), a.incart_dir, "II"),
            "sv": (sorted(p.stem for p in Path(a.svdb_dir).glob("*.hea")), a.svdb_dir, "ECG1")}
    X, Y = {}, {}
    for k, (recs, path, lead) in sets.items():
        d = build_dataset(recs, path, cfg, cache_dir=a.cache, verbose=False, lead=lead)
        f = None
        if a.fine_delta:
            f = build_dataset(recs, path, replace(cfg, delta_mv=a.fine_delta), cache_dir=a.cache,
                              verbose=False, lead=lead)
            assert (f["y"] == d["y"]).all()
        X[k], Y[k] = X_of(d, f), d["y"].astype(int)
    rs = np.random.default_rng(0)
    keep = np.ones(len(Y["inc"]), bool)
    keep[rs.permutation(np.flatnonzero(Y["inc"] == 0))[40000:]] = False
    Xtr = np.concatenate([X["mit"], X["inc"][keep]]); ytr = np.concatenate([Y["mit"], Y["inc"][keep]])
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(Xtr, ytr, sample_weight=np.sqrt(compute_sample_weight("balanced", ytr)))
    r = report(Y["sv"], clf.predict(X["sv"]))
    print(format_report(r, f"== external SVDB (train DS1+INCART), fine-delta {a.fine_delta}"), flush=True)


if __name__ == "__main__":
    main()
