"""Two-stage classifier with causal per-patient prediction memory. DS1 grouped CV only (DS2 never loaded).

Stage 1: GBT on the frozen round-4 features. Stage 2: same features + causal summaries of stage-1
probabilities of *previous* beats (per cluster and overall). Stage-1 probabilities for training beats come
from inner record-grouped OOF fits, so stage 2 never sees in-sample predictions.
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
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402

COLS = list(range(13)) + [21, 22, 23, 24, 25] + [32, 33, 34, 35, 36, 37]


def X_of(d):
    n = len(d["y"])
    return np.concatenate([d["hist"].reshape(n, -1), d["res"].reshape(n, -1),
                           d["scal"][:, COLS]], 1).astype(np.float32)


def clf():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


WPOW = float(__import__("os").environ.get("WPOW", "0.5"))   # class-weight power: 0 none, 0.5 sqrt, 1 balanced


def fitp(X, y, wpow=0.5):
    return clf().fit(X, y, sample_weight=compute_sample_weight("balanced", y) ** wpow)


def memory_feats(P, rec, cid):
    """Causal: per record in time order, EWMA(1/8) of previous stage-1 probs in the same cluster,
    EWMA(1/4) over all previous beats, and the previous beat's probs."""
    out = np.zeros((len(P), 12), np.float32)
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        cm = np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1))
        gm = np.array([1, 0, 0, 0], np.float32)
        prev = gm.copy()
        for i in idx:
            c = int(cid[i])
            out[i, :4] = cm[c]; out[i, 4:8] = gm; out[i, 8:12] = prev
            cm[c] += (P[i] - cm[c]) / 8
            gm += (P[i] - gm) / 4
            prev = P[i]
    return out


def oof(X, y, groups, folds=4):
    P = np.zeros((len(y), 4), np.float32)
    for tr, te in GroupKFold(folds).split(X, y, groups):
        P[te] = fitp(X[tr], y[tr]).predict_proba(X[te])
    return P


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "mitdb")
    INC = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "data" / "incart")
    cache = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "data" / "cache")
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, _ = data.get_splits(D)
    d = build_dataset(ds1, D, cfg, cache_dir=cache, verbose=False)
    recs = sorted(p.stem for p in Path(INC).glob("*.hea"))
    e = build_dataset(recs, INC, cfg, cache_dir=cache, verbose=False, lead="II")
    X, y, g, cid = X_of(d), d["y"].astype(int), d["rec"], d["scal"][:, 38]
    rs = np.random.default_rng(0)
    ye_all = e["y"].astype(int)
    keep = np.ones(len(ye_all), bool)
    keep[rs.permutation(np.flatnonzero(ye_all == 0))[40000:]] = False
    Xe, ye, ge, cide = X_of(e)[keep], ye_all[keep], e["rec"][keep], e["scal"][keep, 38]
    # memory features need the full time order; for INCART the subsample drops some N beats - recompute on kept beats

    p1 = np.zeros_like(y); p2 = np.zeros_like(y); P1o = np.zeros((len(y), 4)); P2o = np.zeros((len(y), 4))
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        Xtr = np.concatenate([X[tr], Xe]); ytr = np.concatenate([y[tr], ye]); gtr = np.concatenate([g[tr], ge])
        Ptr = oof(Xtr, ytr, gtr)                                   # inner OOF stage-1 probs
        m1 = fitp(Xtr, ytr)
        Pte = m1.predict_proba(X[te]).astype(np.float32)
        p1[te] = Pte.argmax(1); P1o[te] = Pte
        Mtr = memory_feats(Ptr, gtr, np.concatenate([cid[tr], cide]))
        Mte = memory_feats(Pte, g[te], cid[te])
        m2 = fitp(np.concatenate([Xtr, Mtr], 1), ytr, WPOW)
        P2o[te] = m2.predict_proba(np.concatenate([X[te], Mte], 1)); p2[te] = P2o[te].argmax(1)
        print(f"fold {k} done", flush=True)
    print(format_report(report(y, p1), "== stage 1 (DS1 grouped CV)"))
    np.savez("/tmp/stage2_oof.npz", P1=P1o, P2=P2o, y=y)
    print(format_report(report(y, p2), f"== stage 2 with causal prediction memory, stage-2 weight power {WPOW} (DS1 grouped CV)"))


def external(D, INC, SV, cache):
    """Train both stages on all of DS1 + INCART, evaluate on SVDB (external dev set)."""
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, _ = data.get_splits(D)
    d = build_dataset(ds1, D, cfg, cache_dir=cache, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=cache, verbose=False, lead="II")
    v = build_dataset(sorted(p.stem for p in Path(SV).glob("*.hea")), SV, cfg, cache_dir=cache, verbose=False, lead="ECG1")
    rs = np.random.default_rng(0)
    ye = e["y"].astype(int); keep = np.ones(len(ye), bool)
    keep[rs.permutation(np.flatnonzero(ye == 0))[40000:]] = False
    Xtr = np.concatenate([X_of(d), X_of(e)[keep]]); ytr = np.concatenate([d["y"].astype(int), ye[keep]])
    gtr = np.concatenate([d["rec"], e["rec"][keep]]); ctr = np.concatenate([d["scal"][:, 38], e["scal"][keep, 38]])
    Ptr = oof(Xtr, ytr, gtr)
    m1 = fitp(Xtr, ytr)
    m2 = fitp(np.concatenate([Xtr, memory_feats(Ptr, gtr, ctr)], 1), ytr, WPOW)
    Xv, yv = X_of(v), v["y"].astype(int)
    Pv1 = m1.predict_proba(Xv).astype(np.float32)
    Pv2 = m2.predict_proba(np.concatenate([Xv, memory_feats(Pv1, v["rec"], v["scal"][:, 38])], 1))
    np.savez("/tmp/stage2_svdb.npz", P1=Pv1, P2=Pv2, y=yv, g=v["rec"])
    print(format_report(report(yv, Pv1.argmax(1)), "== SVDB external, stage 1"))
    print(format_report(report(yv, Pv2.argmax(1)), "== SVDB external, stage 2"))


if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[4] == "external":
        external(sys.argv[1], sys.argv[2], "/home/claude/data/svdb", sys.argv[3])
        sys.exit()
    main()
