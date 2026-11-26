"""Smallest AF detector that keeps the DS1 dev benefit (rule: DS1-LOPO flag accuracy within 0.02 pt of the 60x15
detector). Dev only: round-4 DS1 LOPO predictions + detector OOF over DS1 records. DS2 never used."""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.metrics import report  # noqa: E402
from scripts.af_detector import afdb_records, mit_records, p_af, rr_features, weights  # noqa: E402
from scripts.final_gbt import footprint_bytes  # noqa: E402

D, AF, OOF = sys.argv[1:4]
z = np.load(OOF); y1, p1, g1, r1 = z["y1"], z["P1"].argmax(1), z["g1"], z["r1"]
ds1, _ = data.get_splits(D)
Xa, ya, _ = afdb_records(AF)
F1 = np.zeros((len(r1), Xa.shape[1]), np.float32)
for rec in np.unique(g1):
    k = np.flatnonzero(g1 == rec); F1[k] = rr_features(r1[k] / 360)
recs = np.array([int(x) for x in ds1])
folds = list(GroupKFold(5).split(recs, recs, recs))
mits = [mit_records(D, [str(x) for x in recs[tr]]) for tr, _ in folds]
for it, lf in ((60, 15), (30, 8), (20, 8), (20, 4), (10, 4), (5, 4)):
    mk = lambda: HistGradientBoostingClassifier(max_iter=it, learning_rate=max(0.1, 3.0 / it), max_leaf_nodes=lf, random_state=0)
    paf = np.zeros(len(y1))
    for (tr, te), (Xm, ym, _) in zip(folds, mits):
        X = np.concatenate([Xa, Xm]); yy = np.concatenate([ya, ym]); w = weights(len(ya), len(ym)); ok = ~np.isnan(X).any(1)
        m = mk().fit(X[ok], yy[ok], sample_weight=w[ok])
        rows = np.isin(g1, recs[te]); paf[rows] = p_af(m, F1[rows])
    q = p1.copy(); q[(q == 1) & (paf > 0.5)] = 0
    r = report(y1, q)
    print(f"iters {it:2d} leaves {lf:2d}  {footprint_bytes(m)['bytes']:5d} B  DS1 LOPO acc {100 * r['accuracy']:.2f}  S F1 {r['S']['f1']:.3f}", flush=True)
