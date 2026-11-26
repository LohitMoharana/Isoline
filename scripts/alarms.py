"""Episode-level alarm logic on top of the frozen 5.2 KB model (round 7).

Rules (cheap, causal, run once per beat / per minute on the device):
  1. Irregular-rhythm flag: irr = median|dRR| / median RR over the previous 16 beats. If irr > T the beat's S
     prediction is reported as N (AF beats are labelled N; irregular timing makes them look premature).
  2. Episode reporting: each record is cut into 60-s windows; a window raises a V (S) alarm if it has >= m_V (m_S)
     predicted V (S) beats.
Window-level truth: a window is a true V (S) episode if it contains >= 1 annotated V (S) beat.
A false alarm is an alarm window with no annotated beat of that class.

m_V, m_S and T are chosen on DS1 only (leave-one-patient-out predictions of the same model recipe), by the rule
fixed here: maximise window-level F1 per class. They are then frozen and applied to DS2, other databases and
noisy DS2. Nothing is tuned on those.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

os.environ["TINY"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final_gbt as F  # noqa: E402

FS = 360
WIN_S = 60
M_GRID = [1, 2, 3, 4, 6, 8]
T_GRID = [None, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]


def irregularity(r, rec):
    out = np.zeros(len(r))
    for g in np.unique(rec):
        idx = np.flatnonzero(rec == g)
        rr = np.diff(r[idx]).astype(float)
        for j, i in enumerate(idx):
            h = rr[max(j - 17, 0):max(j - 1, 0)]                 # previous RR intervals only (causal)
            if len(h) >= 8:
                out[i] = np.median(np.abs(np.diff(h))) / max(np.median(h), 1)
    return out


def apply_af(p, irr, T):
    q = p.copy()
    if T is not None:
        q[(q == 1) & (irr > T)] = 0
    return q


def windows(y, p, r, rec, cls, m):
    tp = fn = fp = 0
    hours = 0.0
    for g in np.unique(rec):
        k = rec == g
        w = (r[k] // (WIN_S * FS)).astype(int)
        nw = int(w.max()) + 1
        hours += nw * WIN_S / 3600
        true = np.bincount(w[y[k] == cls], minlength=nw) >= 1
        pred = np.bincount(w[p[k] == cls], minlength=nw) >= m
        tp += int((true & pred).sum()); fn += int((true & ~pred).sum()); fp += int((~true & pred).sum())
    se = tp / max(tp + fn, 1); ppv = tp / max(tp + fp, 1)
    return dict(se=se, ppv=ppv, f1=2 * se * ppv / max(se + ppv, 1e-9), tp=tp, fn=fn, fp=fp,
                false_alarms_per_hour=fp / max(hours, 1e-9), hours=hours)


def summarize(name, y, p, r, rec, cfg_rules, irr):
    q = apply_af(p, irr, cfg_rules["T"])
    beat_raw, beat_af = report(y, p), report(y, q)
    out = dict(beat_raw=beat_raw, beat_after_af_flag=beat_af)
    line = [f"{name:16s} beat acc {100 * beat_raw['accuracy']:.2f} -> {100 * beat_af['accuracy']:.2f} (AF flag)"]
    for cls, key, m in ((2, "V", cfg_rules["m_V"]), (1, "S", cfg_rules["m_S"])):
        raw = windows(y, p, r, rec, cls, 1)
        smo = windows(y, q, r, rec, cls, m)
        out[f"{key}_windows_raw"], out[f"{key}_windows_alarm"] = raw, smo
        line.append(f"  {key}: per-beat alarms Se {raw['se']:.2f} +P {raw['ppv']:.2f} FA/h {raw['false_alarms_per_hour']:.1f}"
                    f"  ->  episode alarms Se {smo['se']:.2f} +P {smo['ppv']:.2f} FA/h {smo['false_alarms_per_hour']:.1f}")
    print("\n".join(line), flush=True)
    return out


def predict_db(model, path, recs, lead, cfg, cache, batch=3):
    ys, ps, rs, gs = [], [], [], []
    for k in range(0, len(recs), batch):
        d = build_dataset(recs[k:k + batch], path, cfg, cache_dir=cache, verbose=False, lead=lead)
        ys.append(d["y"].astype(int)); ps.append(model.predict(F.X_of(d))); rs.append(d["r"]); gs.append(d["rec"])
        del d
    return [np.concatenate(a) for a in (ys, ps, rs, gs)]


def main():
    D, INC, DATA, NST, C, OUT = sys.argv[1:7]
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, ds2 = data.get_splits(D)
    d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
    X1, y1, g1, r1 = F.X_of(d1), d1["y"].astype(int), d1["rec"], d1["r"]
    Xe, ye = F.X_of(e), e["y"].astype(int)

    # ---- dev: DS1 leave-one-patient-out predictions, choose rules
    p1 = np.zeros_like(y1)
    for tr, te in GroupKFold(len(np.unique(g1))).split(X1, y1, g1):
        p1[te] = F.fit(X1[tr], y1[tr], Xe, ye, 0).predict(X1[te])
    irr1 = irregularity(r1, g1)
    best_v = max(M_GRID, key=lambda m: (windows(y1, p1, r1, g1, 2, m)["f1"], -m))
    best_s = max(((m, T) for m in M_GRID for T in T_GRID),
                 key=lambda mt: (windows(y1, apply_af(p1, irr1, mt[1]), r1, g1, 1, mt[0])["f1"], -mt[0]))
    rules = dict(m_V=best_v, m_S=best_s[0], T=best_s[1])
    print("rules chosen on DS1 LOPO:", rules, flush=True)
    res = {"rules": rules, "DS1_LOPO_dev": summarize("DS1 LOPO (dev)", y1, p1, r1, g1, rules, irr1)}

    # ---- frozen model and rules, test sets
    model = F.fit(X1, y1, Xe, ye, 0)
    tests = [("DS2", D, ds2, "MLII")] + [(db.upper(), str(Path(DATA) / db), sorted(p.stem for p in Path(DATA, db).glob("*.hea")), lead)
                                         for db, lead in (("svdb", "ECG1"), ("edb", "ML"), ("ltdb", "ECG1"), ("nsrdb", "ECG1"))]
    for name, path, recs, lead in tests:
        y, p, r, g = predict_db(model, path, recs, lead, cfg, C)
        res[name] = summarize(name, y, p, r, g, rules, irregularity(r, g))
        Path(OUT, "alarms.json").write_text(json.dumps(res, indent=2, default=float))

    # ---- noisy DS2 (electrode motion, the worst case)
    import wfdb
    import scripts.noise_stress as N
    em = wfdb.rdrecord(str(Path(NST) / "em")).p_signal[:, 0]
    for snr in (12, 6):
        data.load_mitdb_record = N.make_noisy_loader(em, snr)
        d2 = build_dataset(ds2, D, cfg, cache_dir=None, verbose=False)
        y, p = d2["y"].astype(int), model.predict(F.X_of(d2))
        res[f"DS2_em{snr}dB"] = summarize(f"DS2 em {snr} dB", y, p, d2["r"], d2["rec"], rules, irregularity(d2["r"], d2["rec"]))
        data.load_mitdb_record = N._clean_loader
        Path(OUT, "alarms.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
