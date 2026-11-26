"""Frozen round 11 (chosen on DS1 + SVDB only): beat classifier = round-4 features + robust RR features
(scripts/exp_robust_rr.py); PRIMARY = with next-RR/dominant (one-beat delay), SECONDARY = causal. AF flag = 95-byte
AF detector (5 trees, 4 leaves; scripts/af_shrink.py) trained on AFDB + DS1 rhythm labels. Both the round-4-size
model and the 5.2 KB tiny model. One DS2 evaluation (12th in the project), 5 seeds."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final_gbt as F  # noqa: E402
from scripts.af_detector import afdb_records, mit_records, p_af, rr_features, weights  # noqa: E402
from scripts.exp_robust_rr import robust  # noqa: E402

D, INC, AF, C, OUT = sys.argv[1:6]
cfg = replace(AFEConfig(), delta_mv=0.1)
ds1, ds2 = data.get_splits(D)
Xa, ya, _ = afdb_records(AF); Xm, ym, _ = mit_records(D, ds1)
Xd = np.concatenate([Xa, Xm]); yd = np.concatenate([ya, ym]); w = weights(len(ya), len(ym)); ok = ~np.isnan(Xd).any(1)
det = HistGradientBoostingClassifier(max_iter=5, learning_rate=0.6, max_leaf_nodes=4, random_state=0).fit(Xd[ok], yd[ok], sample_weight=w[ok])
d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
d2 = build_dataset(ds2, D, cfg, cache_dir=C, verbose=False)
e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
y1, ye, y2 = d1["y"].astype(int), e["y"].astype(int), d2["y"].astype(int)
Fr = np.zeros((len(y2), Xa.shape[1]), np.float32)
for rec in np.unique(d2["rec"]):
    k = np.flatnonzero(d2["rec"] == rec); Fr[k] = rr_features(d2["r"][k] / 360)
af = p_af(det, Fr) > 0.5
out = {"af_detector_bytes": F.footprint_bytes(det)["bytes"]}
for variant, delayed in (("primary_delayed", True), ("secondary_causal", False)):
    X1 = np.concatenate([F.X_of(d1), robust(d1, delayed)], 1)
    Xe = np.concatenate([F.X_of(e), robust(e, delayed)], 1)
    X2 = np.concatenate([F.X_of(d2), robust(d2, delayed)], 1)
    for size, tiny in (("round4", False), ("tiny", True)):
        F.TINY = tiny
        runs = []
        for s in range(5):
            m = F.fit(X1, y1, Xe, ye, s)
            p = m.predict(X2); q = p.copy(); q[(q == 1) & af] = 0
            r = report(y2, q); r["no_flag"] = report(y2, p)["accuracy"]; r["bytes"] = F.footprint_bytes(m)["bytes"]; runs.append(r)
        summ = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in ("accuracy", "macro_f1", "no_flag", "bytes")}
        for c in "NSVF":
            for mt in ("f1", "se", "ppv"):
                v = [r[c][mt] for r in runs]; summ[f"{mt}_{c}"] = (float(np.mean(v)), float(np.std(v)))
        key = f"{variant}_{size}"
        out[key] = dict(summary=summ, runs=runs)
        print(key, {k: [round(x, 4) for x in v] for k, v in summ.items()}, flush=True)
        Path(OUT, "final_round11.json").write_text(json.dumps(out, indent=2, default=float))
