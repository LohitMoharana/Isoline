"""Frozen round-10: round-4 model and 5.2 KB model, each with the AF-detector flag (P(AF) > 0.5 -> S reported as N).
The detector is trained on AFDB + all DS1 rhythm labels. One DS2 evaluation (11th in the project), 5 seeds per model."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final.final_gbt as F  # noqa: E402
from scripts.final.af_detector import afdb_records, mit_records, p_af, rr_features, train  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

D, INC, AF, C, OUT = sys.argv[1:6]
cfg = replace(AFEConfig(), delta_mv=0.1)
ds1, ds2 = data.get_splits(D)
Xa, ya, _ = afdb_records(AF); Xm, ym, _ = mit_records(D, ds1)
det = train(Xa, ya, Xm, ym)
d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
d2 = build_dataset(ds2, D, cfg, cache_dir=C, verbose=False)
e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
X1, y1 = F.X_of(d1), d1["y"].astype(int); Xe, ye = F.X_of(e), e["y"].astype(int)
X2, y2 = F.X_of(d2), d2["y"].astype(int)
Fr = np.zeros((len(y2), Xa.shape[1]), np.float32)
for rec in np.unique(d2["rec"]):
    k = np.flatnonzero(d2["rec"] == rec); Fr[k] = rr_features(d2["r"][k] / 360)
af = p_af(det, Fr) > 0.5
out = {}
for name, tiny in (("round4", False), ("tiny_5KB", True)):
    F.TINY = tiny
    runs = []
    for s in range(5):
        p = F.fit(X1, y1, Xe, ye, s).predict(X2)
        q = p.copy(); q[(q == 1) & af] = 0
        r = report(y2, q); r["no_flag"] = report(y2, p)["accuracy"]; runs.append(r)
        print(format_report(r, f"== {name} seed {s} AF-detector flag (no flag {100 * r['no_flag']:.2f}%)"), flush=True)
    summ = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in ("accuracy", "macro_f1", "no_flag")}
    for c in "NSVF":
        for m in ("f1", "se", "ppv"):
            v = [r[c][m] for r in runs]; summ[f"{m}_{c}"] = (float(np.mean(v)), float(np.std(v)))
    print(name, {k: [round(x, 4) for x in v] for k, v in summ.items()}, flush=True)
    out[name] = dict(summary=summ, runs=runs)
out["af_flagged_fraction_ds2"] = float(af.mean())
Path(OUT, "final_af_detector.json").write_text(json.dumps(out, indent=2, default=float))
