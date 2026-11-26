"""Frozen DS2 evaluation of option 1 (S-biased operating point). DS2 look #14; b fixed on DS1 OOF (exp_sbias.py):
full b = 20, tiny b = 8. Deployed config (HPF 0.05 Hz), round-10 recipe, AF detector on AFDB + all DS1, 5 seeds.
Baseline (b = 1) is re-measured in the same run so the comparison is paired."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final_gbt as F  # noqa: E402
from scripts.af_detector import afdb_records, mit_records, p_af, rr_features, train  # noqa: E402
from scripts.find_nonpremature_s import nonprem_mask  # noqa: E402

B = {"full": 20.0, "tiny": 8.0}
D, INC, AF, C, OUT = sys.argv[1:6]
cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)
ds1, ds2 = data.get_splits(D)
d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
d2 = build_dataset(ds2, D, cfg, cache_dir=C, verbose=False)
Xa, ya, _ = afdb_records(AF); Xm, ym, _ = mit_records(D, ds1); det = train(Xa, ya, Xm, ym)
X1, y1, Xe, ye, X2, y2 = F.X_of(d1), d1["y"].astype(int), F.X_of(e), e["y"].astype(int), F.X_of(d2), d2["y"].astype(int)
Fr = np.zeros((len(y2), Xa.shape[1]), np.float32)
for rec in np.unique(d2["rec"]):
    k = np.flatnonzero(d2["rec"] == rec); Fr[k] = rr_features(d2["r"][k] / 360)
af = p_af(det, Fr) > 0.5
npm = nonprem_mask(d2); r232 = d2["rec"] == 232
out = {}
for size, tiny in (("full", False), ("tiny", True)):
    F.TINY = tiny
    runs = {"b=1": [], f"b={B[size]:g}": []}
    for s in range(5):
        P = F.fit(X1, y1, Xe, ye, s).predict_proba(X2)
        for key, b in (("b=1", 1.0), (f"b={B[size]:g}", B[size])):
            p = np.argmax(P * np.array([1, b, 1, 1]), 1); p[(p == 1) & af] = 0
            r = report(y2, p)
            r["nonprem_S_se"] = float(((p == 1) & npm).sum() / npm.sum())
            r["S_se_excl_232"] = float(((p == 1) & (y2 == 1) & ~r232).sum() / ((y2 == 1) & ~r232).sum())
            runs[key].append(r)
    out[size] = {}
    for key, rs in runs.items():
        m = lambda f: (float(np.mean([f(r) for r in rs])), float(np.std([f(r) for r in rs])))
        summ = {"accuracy": m(lambda r: r["accuracy"]), "macro_f1": m(lambda r: r["macro_f1"]),
                "S_se": m(lambda r: r["S"]["se"]), "S_ppv": m(lambda r: r["S"]["ppv"]), "S_f1": m(lambda r: r["S"]["f1"]),
                "V_se": m(lambda r: r["V"]["se"]), "V_ppv": m(lambda r: r["V"]["ppv"]), "N_se": m(lambda r: r["N"]["se"]),
                "nonprem_S_se": m(lambda r: r["nonprem_S_se"]), "S_se_excl_232": m(lambda r: r["S_se_excl_232"])}
        out[size][key] = summ
        print(f"DS2 {size} {key:5s}: acc {100 * summ['accuracy'][0]:.2f}+-{100 * summ['accuracy'][1]:.2f}  "
              f"macroF1 {summ['macro_f1'][0]:.3f}  S Se {summ['S_se'][0]:.3f} +P {summ['S_ppv'][0]:.3f} "
              f"F1 {summ['S_f1'][0]:.3f}  S Se excl. 232 {summ['S_se_excl_232'][0]:.3f}  nonprem-S Se "
              f"{summ['nonprem_S_se'][0]:.3f}  V Se {summ['V_se'][0]:.3f} +P {summ['V_ppv'][0]:.3f}  N Se {summ['N_se'][0]:.3f}",
              flush=True)
Path(OUT, "final_sbias.json").write_text(json.dumps(out, indent=2))
