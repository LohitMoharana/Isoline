"""Which mix of extra TRAINING databases helps? DS1 grouped 5-fold CV only (DS2 never loaded)."""
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
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402


def X_of(ds, ncol=13):
    n = len(ds["y"])
    return np.concatenate([ds["hist"].reshape(n, -1), ds["res"].reshape(n, -1),
                           ds["scal"][:, :ncol]], 1).astype(np.float32)


def load(path, lead, cfg, cache, n_keep, seed=0):
    recs = sorted(p.stem for p in Path(path).glob("*.hea"))
    d = build_dataset(recs, path, cfg, cache_dir=cache, verbose=False, lead=lead)
    X, y = X_of(d), d["y"].astype(int)
    if n_keep:
        rs = np.random.default_rng(seed)
        nidx = np.flatnonzero(y == 0)
        drop = rs.permutation(nidx)[n_keep:]
        k = np.ones(len(y), bool); k[drop] = False
        X, y = X[k], y[k]
    print(f"{Path(path).name}: {len(y)} beats {np.bincount(y, minlength=4).tolist()}", flush=True)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mit-dir", required=True)
    ap.add_argument("--svdb-dir", default="/home/claude/data/svdb")
    ap.add_argument("--incart-dir", default="/home/claude/data/incart")
    ap.add_argument("--edb-dir", default="/home/claude/data/edb")
    ap.add_argument("--cache", default="/tmp/cache")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--n-keep", type=int, default=40000)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--combos", default="none,incart,incart+svdb,incart+edb,incart+svdb+edb")
    a = ap.parse_args()
    cfg = replace(AFEConfig(), delta_mv=a.delta)
    ds1, _ = data.get_splits(a.mit_dir)
    d = build_dataset(ds1, a.mit_dir, cfg, cache_dir=a.cache, verbose=False)
    X, y, g = X_of(d), d["y"].astype(int), d["rec"]
    ex = {}
    need = set(a.combos.replace(",", "+").split("+")) - {"none"}
    if "incart" in need: ex["incart"] = load(a.incart_dir, "II", cfg, a.cache, a.n_keep)
    if "svdb" in need: ex["svdb"] = load(a.svdb_dir, "ECG1", cfg, a.cache, a.n_keep)
    if "edb" in need: ex["edb"] = load(a.edb_dir, "FIRST", cfg, a.cache, a.n_keep)
    for combo in a.combos.split(","):
        names = [] if combo == "none" else combo.split("+")
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, g):
            Xtr = np.concatenate([X[tr]] + [ex[k][0] for k in names])
            ytr = np.concatenate([y[tr]] + [ex[k][1] for k in names])
            w = np.sqrt(compute_sample_weight("balanced", ytr))
            clf = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=a.lr, max_leaf_nodes=a.leaves,
                                                 l2_regularization=1.0, random_state=0).fit(Xtr, ytr, sample_weight=w)
            pred[te] = clf.predict(X[te])
        r = report(y, pred)
        print(f"{combo:14s} it{a.iters} lf{a.leaves} lr{a.lr} acc {100 * r['accuracy']:.2f}  macroF1 {r['macro_f1']:.3f}  S F1 {r['S']['f1']:.3f} "
              f"(Se {r['S']['se']:.2f} +P {r['S']['ppv']:.2f})  V F1 {r['V']['f1']:.3f}  F F1 {r['F']['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
