"""Clean-protocol re-implementation of the random-forest method of Saenz-Cogollo & Agelli,
"Investigating Feature Selection and Random Forests for Inter-Patient Heartbeat Classification",
Algorithms 13(4):75, 2020 (reported 96.14 % on DS2).

What follows the paper (as far as its text states): MLII only, resampled 360 -> 150 Hz, beat segment
56 samples before / 40 after the R annotation with the segment mean subtracted, RR features normalised by
the average of the previous 32 beats, a t-statistic over the last 32 RR intervals, R(+1) (one-beat delay),
QRS widths at fixed fractions of the peak (raw and normalised by the previous-32 average), Hermite-basis
(HBF) coefficients, features ranked by mutual information on DS1, scikit-learn RandomForest with 40 trees.

What is approximated (not specified in the text available to us): exact width fractions (we use 0.2, 0.4,
0.6, 0.8), HBF degree/width (degree 0..4, sigma 15 ms over +/-100 ms), and the exact feature set.

Protocol difference (the point of this script): the number of features k is chosen by leave-one-patient-out
CV on DS1, NOT on DS2 as in the paper. For transparency the DS2 accuracy at every k is also printed, labelled
as the biased "k chosen on DS2" protocol; it is never used to choose anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AAMI_MAP, CLASS_IDX  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402

FS2 = 150
PRE2, POST2 = 56, 40
QRS = 15                                     # +/-100 ms at 150 Hz
FRACS = (0.2, 0.4, 0.6, 0.8)


def hermite_basis(n_max=4, half=QRS, sigma_s=0.015):
    t = np.arange(-half, half + 1) / FS2
    x = t / sigma_s
    H = [np.ones_like(x), 2 * x]
    for n in range(2, n_max + 1):
        H.append(2 * x * H[-1] - 2 * (n - 1) * H[-2])
    B = np.stack([h * np.exp(-x ** 2 / 2) for h in H[: n_max + 1]], 1)
    return B / np.linalg.norm(B, axis=0)


HB = hermite_basis()


def record_features(D, rec):
    sig, r, sym = data.load_mitdb_record(D, rec)
    x = resample_poly(sig, 5, 12)
    r2 = np.round(np.asarray(r) * FS2 / 360).astype(int)
    rr = np.diff(r2).astype(float)
    feats, ys = [], []
    hist_rr, hist_w = [], []
    for i in range(1, len(r2) - 1):
        c = AAMI_MAP.get(sym[i])
        R0, Rm1, Rp1 = rr[i - 1], (rr[i - 2] if i >= 2 else rr[i - 1]), rr[i]
        prev = np.array(hist_rr[-32:]) if hist_rr else np.array([R0])
        avg, sd = prev.mean(), prev.std() + 1e-6
        s, e = r2[i] - PRE2, r2[i] + POST2
        ok = s >= 0 and e <= len(x)
        if ok:
            seg = x[s:e] - x[s:e].mean()
            q = seg[PRE2 - QRS:PRE2 + QRS + 1]
            pk = np.argmax(np.abs(q)); amp = q[pk]
            w = [float((q * np.sign(amp) >= f * abs(amp)).sum()) for f in FRACS]
            pw = np.array(hist_w[-32:]) if hist_w else np.array([w])
            wn = list(np.array(w) / np.maximum(pw.mean(0), 1e-6))
            hbf = list(HB.T @ q)
            hist_w.append(w)
        hist_rr.append(R0)
        if not ok or c is None or c == "Q":
            continue
        feats.append([R0, Rm1, Rp1, avg, R0 / avg, Rm1 / R0, Rp1 / R0, avg / R0, (R0 - avg) / sd]
                     + w + wn + hbf + [abs(amp)])
        ys.append(CLASS_IDX[c])
    return np.array(feats, np.float32), np.array(ys, int)


NAMES = (["R0", "R-1", "R+1", "avgRR32", "R0/avg", "R-1/R0", "R+1/R0", "avg/R0", "t_stat"]
         + [f"w{int(f * 10)}" for f in FRACS] + [f"w{int(f * 10)}_n" for f in FRACS]
         + [f"hbf_c{j}" for j in range(HB.shape[1])] + ["amp"])


def build(D, recs):
    X, Y, G = [], [], []
    for rec in recs:
        f, y = record_features(D, rec)
        X.append(f); Y.append(y); G.append(np.full(len(y), int(rec)))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def rf(seed):
    return RandomForestClassifier(n_estimators=40, random_state=seed, n_jobs=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    ds1, ds2 = data.get_splits(a.data_dir)
    X1, y1, g1 = build(a.data_dir, ds1)
    mi = mutual_info_classif(X1, y1, random_state=0)
    order = np.argsort(-mi)
    print("MI ranking on DS1:", [NAMES[j] for j in order[: a.kmax]], flush=True)
    cv = {}
    for k in range(1, a.kmax + 1):
        cols = order[:k]; pred = np.zeros_like(y1)
        for tr, te in GroupKFold(len(np.unique(g1))).split(X1, y1, g1):      # leave-one-patient-out
            pred[te] = rf(0).fit(X1[tr][:, cols], y1[tr]).predict(X1[te][:, cols])
        cv[k] = report(y1, pred)["accuracy"]
        print(f"k={k:2d}  DS1 LOPO accuracy {100 * cv[k]:.2f}", flush=True)
    k_star = max(cv, key=lambda k: (cv[k], -k))
    print(f"chosen on DS1: k = {k_star}  ({[NAMES[j] for j in order[:k_star]]})")
    out = dict(mi_order=[NAMES[j] for j in order], ds1_lopo=cv, k_star=int(k_star))
    if a.final:
        X2, y2, _ = build(a.data_dir, ds2)
        runs = []
        for s in range(5):
            r = report(y2, rf(s).fit(X1[:, order[:k_star]], y1).predict(X2[:, order[:k_star]]))
            runs.append(r)
            print(format_report(r, f"== DS2 clean protocol, k={k_star}, seed {s}"), flush=True)
        out["ds2_clean"] = runs
        accs = [r["accuracy"] for r in runs]
        print(f"\nDS2 clean protocol: accuracy {100 * np.mean(accs):.2f} +/- {100 * np.std(accs):.2f}")
        biased = {}
        for k in range(1, a.kmax + 1):
            biased[k] = report(y2, rf(0).fit(X1[:, order[:k]], y1).predict(X2[:, order[:k]]))["accuracy"]
        kb = max(biased, key=biased.get)
        print("illustration only, paper-style 'k chosen on DS2':",
              {k: round(100 * v, 2) for k, v in biased.items()}, f"-> would report {100 * biased[kb]:.2f} at k={kb}")
        out["ds2_by_k_biased_illustration"] = biased
    (Path(a.out) / "replicate_rf.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
