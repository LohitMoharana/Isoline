"""End-to-end evaluation: beats detected from comparator events (isoline.qrs, parameters chosen on DS1) instead of
annotated R peaks, then the round-10 classifier (trained as before on DS1 + INCART at annotated positions, Python AFE
at the circuit's 0.05 Hz high-pass). EC57-style: detections are matched to annotations within 150 ms; classification
is scored on matched beats; missed beats are also reported as errors in an end-to-end accuracy."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data, pipeline, qrs  # noqa: E402
from isoline.afe import run_afe  # noqa: E402
from isoline.config import AAMI_MAP, AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final_gbt as F  # noqa: E402
from scripts.af_detector import afdb_records, mit_records, p_af, rr_features, train  # noqa: E402

D, INC, AF, C, OUT = sys.argv[1:6]
recs_split = sys.argv[6] if len(sys.argv) > 6 else "DS2"
cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)
ds1, ds2 = data.get_splits(D)
d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
Xa, ya, _ = afdb_records(AF); Xm, ym, _ = mit_records(D, ds1); det_af = train(Xa, ya, Xm, ym)
X1, y1, Xe, ye = F.X_of(d1), d1["y"].astype(int), F.X_of(e), e["y"].astype(int)
recs = ds2 if recs_split == "DS2" else ds1

_load = data.load_mitdb_record
stats = {"tp": 0, "fn": 0, "fp": 0, "missed_by_class": {c: 0 for c in "NSVFQ"}, "ref_by_class": {c: 0 for c in "NSVFQ"}}


def detected_loader(data_dir, rec, lead="MLII"):
    x, r, sym = _load(data_dir, rec, lead=lead)
    up, dn, _ = run_afe(x, cfg, np.random.default_rng(0))
    det = qrs.detect(up, dn)
    ref = np.asarray(r)
    tp, fn, fp, m = qrs.match(det, ref)
    stats["tp"] += tp; stats["fn"] += fn; stats["fp"] += fp
    for i, s in enumerate(sym):
        c = AAMI_MAP.get(s)
        if c:
            stats["ref_by_class"][c] += 1
            if m[i] < 0:
                stats["missed_by_class"][c] += 1
    lab = np.array(["~"] * len(det), dtype=object)                   # unmatched detections: no label (dropped)
    for i, k in enumerate(m):
        if k >= 0:
            lab[k] = sym[i]
    return x, det, list(lab)


pipeline.data.load_mitdb_record = detected_loader
d2 = build_dataset(recs, D, cfg, cache_dir=None, verbose=False)
pipeline.data.load_mitdb_record = _load
Fr = np.zeros((len(d2["y"]), Xa.shape[1]), np.float32)
for rec in np.unique(d2["rec"]):
    k = np.flatnonzero(d2["rec"] == rec); Fr[k] = rr_features(d2["r"][k] / 360)
af = p_af(det_af, Fr) > 0.5
X2, y2 = F.X_of(d2), d2["y"].astype(int)
n_ref = sum(stats["ref_by_class"][c] for c in "NSVF"); n_missed = sum(stats["missed_by_class"][c] for c in "NSVF")
out = {"detection": {"Se": stats["tp"] / (stats["tp"] + stats["fn"]), "PPV": stats["tp"] / (stats["tp"] + stats["fp"]),
                     **stats}}
for size, tiny in (("full", False), ("tiny", True)):
    F.TINY = tiny
    runs = []
    for s in range(5):
        p = F.fit(X1, y1, Xe, ye, s).predict(X2); p[(p == 1) & af] = 0
        r = report(y2, p); r["end_to_end_accuracy"] = r["accuracy"] * len(y2) / (len(y2) + n_missed); runs.append(r)
    summ = {k: float(np.mean([r[k] for r in runs])) for k in ("accuracy", "end_to_end_accuracy", "macro_f1")}
    summ.update({f"f1_{c}": float(np.mean([r[c]["f1"] for r in runs])) for c in "NSVF"})
    out[size] = summ
    print(format_report(runs[0], f"== {recs_split} {size}, detected beats (seed 0)"))
    print(f"{size}: matched-beat accuracy {100 * summ['accuracy']:.2f} %, end-to-end (missed beats = errors) "
          f"{100 * summ['end_to_end_accuracy']:.2f} %, S F1 {summ['f1_S']:.3f}, V F1 {summ['f1_V']:.3f}", flush=True)
print(f"beat detection on {recs_split}: Se {100 * out['detection']['Se']:.2f} %  +P {100 * out['detection']['PPV']:.2f} %  "
      f"missed by class {stats['missed_by_class']}")
Path(OUT, f"end_to_end_{recs_split}.json").write_text(json.dumps(out, indent=2))
