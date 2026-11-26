"""Train on features computed at *detected* beat positions (what the device sees) instead of annotated R peaks.

dev (default): DS1 5-fold patient-grouped CV, both variants scored on DS1 detected beats:
   (a) trained on annotated-position features (current), (b) trained on detected-position features.
   Rule: adopt (b) only if its DS1 accuracy is higher.
--final: train (b) on DS1 + INCART detected beats and evaluate end to end on DS2 detected beats (5 seeds).
Labels of detected beats come from the annotation matched within 150 ms; unmatched detections are dropped from
training and counted as false detections in evaluation.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

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
FINAL = "--final" in sys.argv
cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)
_load = data.load_mitdb_record
STATS = {}


def detected_loader(data_dir, rec, lead="MLII"):
    x, r, sym = _load(data_dir, rec, lead=lead)
    up, dn, _ = run_afe(x, cfg, np.random.default_rng(0))
    det = qrs.detect(up, dn); ref = np.asarray(r)
    tp, fn, fp, m = qrs.match(det, ref)
    st = STATS.setdefault(Path(data_dir).name, dict(tp=0, fn=0, fp=0, missed=0))
    st["tp"] += tp; st["fn"] += fn; st["fp"] += fp
    st["missed"] += sum(1 for i, s in enumerate(sym) if AAMI_MAP.get(s) in ("N", "S", "V", "F") and m[i] < 0)
    lab = np.array(["~"] * len(det), dtype=object)
    for i, k in enumerate(m):
        if k >= 0:
            lab[k] = sym[i]
    return x, det, list(lab)


def build(recs, path, lead, detected, cache_name):
    cp = Path("/tmp") / f"det_{cache_name}_{int(detected)}.npz"
    if cp.exists():
        z = np.load(cp, allow_pickle=True); return {k: z[k] for k in z.files}
    if detected:
        pipeline.data.load_mitdb_record = detected_loader
    d = build_dataset(recs, path, cfg, cache_dir=None, verbose=False, lead=lead)
    pipeline.data.load_mitdb_record = _load
    d.pop("stats", None)
    np.savez(cp, **d)
    return d


def af_flags(d, det_af, nfeat):
    Fr = np.zeros((len(d["y"]), nfeat), np.float32)
    for rec in np.unique(d["rec"]):
        k = np.flatnonzero(d["rec"] == rec); Fr[k] = rr_features(d["r"][k] / 360)
    return p_af(det_af, Fr) > 0.5


ds1, ds2 = data.get_splits(D)
incs = sorted(p.stem for p in Path(INC).glob("*.hea"))
Xa, ya, _ = afdb_records(AF)
A1, A1d = build(ds1, D, "MLII", False, "ds1"), build(ds1, D, "MLII", True, "ds1")
E, Ed = build(incs, INC, "II", False, "inc"), build(incs, INC, "II", True, "inc")

if not FINAL:
    y, g = A1d["y"].astype(int), A1d["rec"]
    Xd = F.X_of(A1d)
    res = {}
    for name, src in (("(a) trained on annotated positions", A1), ("(b) trained on detected positions", A1d)):
        Xs, ys, gs = F.X_of(src), src["y"].astype(int), src["rec"]
        Xi, yi = (F.X_of(E), E["y"].astype(int)) if src is A1 else (F.X_of(Ed), Ed["y"].astype(int))
        pred = np.zeros_like(y)
        recs = np.unique(g)
        for tr, te in GroupKFold(5).split(recs, recs, recs):
            trm, tem = np.isin(gs, recs[tr]), np.isin(g, recs[te])
            Xm_, ym_, _ = mit_records(D, [str(int(r)) for r in recs[tr]])
            det_af = train(Xa, ya, Xm_, ym_)
            p = F.fit(Xs[trm], ys[trm], Xi, yi, 0).predict(Xd[tem])
            p[(p == 1) & af_flags({k: v[tem] for k, v in A1d.items()}, det_af, Xa.shape[1])] = 0
            pred[tem] = p
        r = report(y, pred); res[name] = r["accuracy"]
        print(f"{name}: DS1 detected-beat accuracy {100 * r['accuracy']:.2f} %  macroF1 {r['macro_f1']:.3f}  "
              f"S F1 {r['S']['f1']:.3f}  V F1 {r['V']['f1']:.3f} (+P {r['V']['ppv']:.3f})", flush=True)
    print("adopt (b):", res["(b) trained on detected positions"] > res["(a) trained on annotated positions"])
else:
    D2 = build(ds2, D, "MLII", True, "ds2")
    Xm_, ym_, _ = mit_records(D, ds1); det_af = train(Xa, ya, Xm_, ym_)
    af = af_flags(D2, det_af, Xa.shape[1])
    X2, y2 = F.X_of(D2), D2["y"].astype(int)
    st = STATS.get(Path(D).name, {})
    out = {"detection_ds2": st}
    for size, tiny in (("full", False), ("tiny", True)):
        F.TINY = tiny
        runs = []
        for s in range(5):
            p = F.fit(F.X_of(A1d), A1d["y"].astype(int), F.X_of(Ed), Ed["y"].astype(int), s).predict(X2)
            p[(p == 1) & af] = 0
            runs.append(report(y2, p))
        acc = float(np.mean([r["accuracy"] for r in runs]))
        missed = st.get("missed", 0)
        e2e = acc * len(y2) / (len(y2) + missed) if st else float("nan")
        out[size] = dict(accuracy=acc, std=float(np.std([r["accuracy"] for r in runs])), end_to_end=e2e,
                         **{f"f1_{c}": float(np.mean([r[c]["f1"] for r in runs])) for c in "NSVF"})
        print(format_report(runs[0], f"== DS2 {size}, trained on detected beats (seed 0)"))
        print(f"{size}: accuracy {100 * acc:.2f} +/- {100 * out[size]['std']:.2f} %, end to end {100 * e2e:.2f} %", flush=True)
    Path(OUT, "end_to_end_detected_training.json").write_text(json.dumps(out, indent=2))
