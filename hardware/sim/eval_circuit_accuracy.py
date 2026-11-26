"""Accuracy of the round-10 system when DS2 passes through the simulated circuit instead of the Python AFE model.

Models: round-4 features + boosted trees (full and 5.2 KB tiny) + RR-only AF detector, trained on DS1 + INCART with
the Python AFE set to the circuit's 0.05 Hz high-pass (the recipe is unchanged, nothing is tuned).
DS2 front ends compared: (a) Python AFE model at the same settings, (b) the ngspice circuit (AC response computed by
ngspice from the netlist + rail clipping; validated against full transient simulation, r = 0.99995).
Comparator offset/hysteresis and electrode noise are not modelled. This is a measurement, not a selection step.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "hardware" / "sim"))
from isoline import afe, data, pipeline  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final.final_gbt as F  # noqa: E402
from scripts.final.af_detector import afdb_records, mit_records, p_af, rr_features, train  # noqa: E402
from spice_frontend import spice_afe_fast  # noqa: E402

D, INC, AF, C, OUT = sys.argv[1:6]
cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)
ds1, ds2 = data.get_splits(D)
d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
Xa, ya, _ = afdb_records(AF); Xm, ym, _ = mit_records(D, ds1); det = train(Xa, ya, Xm, ym)
X1, y1, Xe, ye = F.X_of(d1), d1["y"].astype(int), F.X_of(e), e["y"].astype(int)

fronts = {"python_afe": build_dataset(ds2, D, cfg, cache_dir=C, verbose=False)}
_orig = afe.run_afe
clip = []


def circuit(x, cfg_, rng):
    up, dn, y, c = spice_afe_fast(x, cfg_.delta_mv); clip.append(c); return up, dn, y


pipeline.afe.run_afe = circuit
fronts["ngspice_circuit"] = build_dataset(ds2, D, cfg, cache_dir=None, verbose=False)
pipeline.afe.run_afe = _orig
d2p, d2c = fronts["python_afe"], fronts["ngspice_circuit"]
assert (d2p["y"] == d2c["y"]).all()
Fr = np.zeros((len(d2p["y"]), Xa.shape[1]), np.float32)
for rec in np.unique(d2p["rec"]):
    k = np.flatnonzero(d2p["rec"] == rec); Fr[k] = rr_features(d2p["r"][k] / 360)
af = p_af(det, Fr) > 0.5
out = {"circuit_clip_fraction_mean": float(np.mean(clip)),
       "idle_sparsity": {k: v["stats"]["idle_sparsity"] for k, v in fronts.items()}}
for size, tiny in (("full", False), ("tiny", True)):
    F.TINY = tiny
    for front, d2 in fronts.items():
        X2, y2 = F.X_of(d2), d2["y"].astype(int)
        runs = []
        for s in range(5):
            p = F.fit(X1, y1, Xe, ye, s).predict(X2); p[(p == 1) & af] = 0
            runs.append(report(y2, p))
        acc = [r["accuracy"] for r in runs]
        summ = {"accuracy": (float(np.mean(acc)), float(np.std(acc))),
                **{f"f1_{c}": float(np.mean([r[c]["f1"] for r in runs])) for c in "NSVF"},
                "macro_f1": float(np.mean([r["macro_f1"] for r in runs]))}
        out[f"{size}_{front}"] = summ
        print(f"{size:4s} {front:16s} acc {100 * summ['accuracy'][0]:.2f} +/- {100 * summ['accuracy'][1]:.2f}  "
              f"macroF1 {summ['macro_f1']:.3f}  S {summ['f1_S']:.3f}  V {summ['f1_V']:.3f}  F {summ['f1_F']:.3f}", flush=True)
print("idle sparsity:", out["idle_sparsity"], " circuit clipping fraction:", out["circuit_clip_fraction_mean"])
Path(OUT, "circuit_accuracy.json").write_text(json.dumps(out, indent=2))
