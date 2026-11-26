"""Compare rhythm-flag variants on dev data only (DS1 LOPO round-4 predictions, SVDB external).
(a) no flag, (b) irregularity > 0.08 (round 9), (c) AF detector P(AF) > 0.5 (detector OOF on DS1 by record).
Rule (fixed before running): pick the highest mean of DS1-LOPO and SVDB accuracy."""
import sys
from pathlib import Path

import numpy as np
import wfdb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline.metrics import report  # noqa: E402
from scripts.final.af_detector import afdb_records, make, mit_records, p_af, rhythm_labels, rr_features, weights  # noqa: E402
from scripts.final.alarms import apply_af, irregularity  # noqa: E402
from isoline import data  # noqa: E402

D, AF, OOF = sys.argv[1:4]
z = np.load(OOF)
y1, p1, g1, r1 = z["y1"], z["P1"].argmax(1), z["g1"], z["r1"]
yv, pv, gv, rv = z["yv"], z["Pv"].argmax(1), z["gv"], z["rv"]
ds1, _ = data.get_splits(D)
Xa, ya, _ = afdb_records(AF)


def feats_rows(r, g):
    X = np.zeros((len(r), 10), np.float32)
    for rec in np.unique(g):
        k = np.flatnonzero(g == rec)
        X[k] = rr_features(r[k] / 360)
    return X


F1 = feats_rows(r1, g1)
# detector OOF for DS1 rows: 5 folds over DS1 records, AFDB always in training
paf1 = np.zeros(len(y1))
recs = np.array([int(x) for x in ds1])
for tr, te in GroupKFold(5).split(recs, recs, recs):
    Xm, ym, _ = mit_records(D, [str(x) for x in recs[tr]])
    X = np.concatenate([Xa, Xm]); yy = np.concatenate([ya, ym]); w = weights(len(ya), len(ym)); ok = ~np.isnan(X).any(1)
    m = make().fit(X[ok], yy[ok], sample_weight=w[ok])
    rows = np.isin(g1, recs[te])
    paf1[rows] = p_af(m, F1[rows])
Xm, ym, _ = mit_records(D, ds1)
X = np.concatenate([Xa, Xm]); yy = np.concatenate([ya, ym]); w = weights(len(ya), len(ym)); ok = ~np.isnan(X).any(1)
m_all = make().fit(X[ok], yy[ok], sample_weight=w[ok])
pafv = p_af(m_all, feats_rows(rv, gv))
irr1, irrv = irregularity(r1, g1), irregularity(rv, gv)


def flag(p, mask):
    q = p.copy(); q[(q == 1) & mask] = 0; return q


res = {}
for name, q1, qv in (("(a) no flag", p1, pv),
                     ("(b) irregularity > 0.08", apply_af(p1, irr1, 0.08), apply_af(pv, irrv, 0.08)),
                     ("(c) AF detector > 0.5", flag(p1, paf1 > 0.5), flag(pv, pafv > 0.5))):
    a, b = report(y1, q1), report(yv, qv)
    res[name] = (a["accuracy"] + b["accuracy"]) / 2
    print(f"{name:26s} DS1 LOPO acc {100 * a['accuracy']:.2f} S Se {a['S']['se']:.2f} +P {a['S']['ppv']:.2f} F1 {a['S']['f1']:.3f} | "
          f"SVDB acc {100 * b['accuracy']:.2f} S Se {b['S']['se']:.2f} +P {b['S']['ppv']:.2f} F1 {b['S']['f1']:.3f} | mean {50 * (a['accuracy'] + b['accuracy']):.2f}", flush=True)
print("chosen:", max(res, key=res.get))
