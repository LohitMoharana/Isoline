"""Causal AF/AFL detector from RR intervals only (previous 16 beats), trained on the MIT-BIH Atrial Fibrillation
Database + MIT-BIH DS1 rhythm annotations. DS2 is never used.

Idea: atrial fibrillation is irregular *without* a dominant RR interval; frequent premature beats are irregular but keep
a dominant normal RR. The rhythm flag used in round 9 (irregularity > T) could not tell these apart and hid real S beats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import wfdb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402

AF_RHYTHMS = ("(AFIB", "(AFL")
NAMES = ["irr_med", "irr_mean", "dominance", "frac_short", "frac_long", "cv", "turning", "rr_rel", "dom_p75", "short_long_pairs"]


def rr_features(t_sec: np.ndarray) -> np.ndarray:
    """Per-beat features from the previous 16 RR intervals (causal). t_sec: beat times of one record."""
    rr = np.diff(t_sec)
    out = np.zeros((len(t_sec), len(NAMES)), np.float32)
    for i in range(len(t_sec)):
        h = rr[max(i - 17, 0):max(i - 1, 0)]
        if len(h) < 8:
            out[i] = np.nan
            continue
        med = np.median(h); d = np.abs(np.diff(h))
        tp = np.mean((h[1:-1] - h[:-2]) * (h[1:-1] - h[2:]) > 0) if len(h) >= 3 else 0
        cur = rr[i - 1] / med if i >= 1 else 1
        out[i] = [np.median(d) / med, d.mean() / h.mean(), np.mean(np.abs(h - med) < 0.1 * med),
                  np.mean(h < 0.85 * med), np.mean(h > 1.15 * med), h.std() / h.mean(), tp, cur,
                  np.mean(np.abs(h - np.percentile(h, 75)) < 0.1 * np.percentile(h, 75)),
                  np.mean((h[:-1] < 0.9 * med) & (h[1:] > 1.05 * med))]
    return out


def rhythm_labels(ann: wfdb.Annotation, samples: np.ndarray) -> np.ndarray:
    """1 if the beat lies inside an AFIB/AFL rhythm interval."""
    starts, kinds = [], []
    for s, a in zip(ann.sample, ann.aux_note):
        a = a.replace("\x00", "").strip()
        if a.startswith("("):
            starts.append(s); kinds.append(a)
    if not starts:
        return np.zeros(len(samples), int)
    idx = np.searchsorted(np.array(starts), samples, side="right") - 1
    lab = np.array([k.startswith(AF_RHYTHMS) for k in kinds])
    return np.where(idx >= 0, lab[np.clip(idx, 0, None)], False).astype(int)


def afdb_records(path):
    X, y, g = [], [], []
    for hea in sorted(Path(path).glob("*.hea")):
        rec = str(hea.with_suffix(""))
        try:
            q = wfdb.rdann(rec, "qrs"); a = wfdb.rdann(rec, "atr")
        except Exception:  # noqa: BLE001
            continue
        fs = wfdb.rdheader(rec).fs
        X.append(rr_features(q.sample / fs)); y.append(rhythm_labels(a, q.sample)); g += [f"afdb{hea.stem}"] * len(q.sample)
    return np.concatenate(X), np.concatenate(y), np.array(g)


def mit_records(path, recs):
    X, y, g = [], [], []
    for rec in recs:
        a = wfdb.rdann(str(Path(path) / rec), "atr")
        beats = np.array([s for s, c in zip(a.sample, a.symbol) if c in data.BEAT_SYMBOLS if hasattr(data, "BEAT_SYMBOLS")] or
                         [s for s, c in zip(a.sample, a.symbol) if c.isalpha() or c in "/!"])
        X.append(rr_features(beats / 360)); y.append(rhythm_labels(a, beats)); g += [f"mit{rec}"] * len(beats)
    return np.concatenate(X), np.concatenate(y), np.array(g)


def make():
    return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, max_leaf_nodes=15, random_state=0)


def weights(na, nm):
    """Each database carries equal total weight (AFDB has ~24x more beats than DS1)."""
    return np.concatenate([np.full(na, 0.5 / na), np.full(nm, 0.5 / nm)]) * (na + nm)


def train(Xa, ya, Xm, ym):
    X = np.concatenate([Xa, Xm]); y = np.concatenate([ya, ym]); w = weights(len(ya), len(ym)); ok = ~np.isnan(X).any(1)
    return make().fit(X[ok], y[ok], sample_weight=w[ok])


def p_af(model, X):
    p = np.zeros(len(X))
    ok = ~np.isnan(X).any(1)
    if ok.any():
        p[ok] = model.predict_proba(X[ok])[:, 1]
    return p


def main():
    D, AF = sys.argv[1], sys.argv[2]
    ds1, _ = data.get_splits(D)
    Xa, ya, ga = afdb_records(AF)
    Xm, ym, gm = mit_records(D, ds1)
    print(f"AFDB beats {len(ya)} (AF {ya.mean():.2f}), DS1 beats {len(ym)} (AF {ym.mean():.3f})")
    X = np.concatenate([Xa, Xm]); y = np.concatenate([ya, ym]); g = np.concatenate([ga, gm]); ok = ~np.isnan(X).any(1)
    w = weights(len(ya), len(ym))
    X, y, g, w = X[ok], y[ok], g[ok], w[ok]
    p = np.zeros(len(y))
    for tr, te in GroupKFold(8).split(X, y, g):
        p[te] = make().fit(X[tr], y[tr], sample_weight=w[tr]).predict_proba(X[te])[:, 1]
    pr = p > 0.5
    for name, m in (("all", np.ones(len(y), bool)), ("AFDB", np.char.startswith(g.astype(str), "afdb")),
                    ("DS1", np.char.startswith(g.astype(str), "mit"))):
        tp = (pr & (y == 1) & m).sum(); fn = (~pr & (y == 1) & m).sum(); fp = (pr & (y == 0) & m).sum()
        print(f"{name:5s} AF beat Se {tp / max(tp + fn, 1):.3f}  +P {tp / max(tp + fp, 1):.3f}  "
              f"false AF on non-AF beats {fp / max(((y == 0) & m).sum(), 1):.4f}")


if __name__ == "__main__":
    main()
