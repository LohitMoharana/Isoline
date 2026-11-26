"""P-wave features, round 2 (dev only; DS2 never loaded).

The earlier attempt (exp_pwave.py) fed raw fine-band histograms to the trees and gained ~0.02 S F1 on DS1, nothing on
SVDB. With only 268 non-premature S beats in DS1 (mostly records 209/207) the trees cannot learn P morphology from
raw bins, so this round uses a few hand-made, patient-normalised P features from the window R-278 ms .. R-61 ms:
  detrended P segment w (line through the means of its first and last 6 samples), then
  positive / negative peak, peak position, correlation and normalised RMS distance to the patient's running P
  template (EWMA 1/16), peak and position relative to the patient's running averages, HF roughness ratio.
All causal (the window ends before R; templates use past beats only).

P sources compared (same features):
  adc      : the conditioned AFE waveform, quantised to the STM32 ADC LSB (3 V / 4096 / gain 300 = 2.44 uV input)
             - the hardware already routes the AFE output to A0, so this costs one conversion per slot
  fine0.05 : a second level-crossing band at 0.05 mV (staircase)
  fine0.025: 0.025 mV
Protocol as exp_nonprem.py (E1: train DS1 + INCART): DS1 grouped 5-fold, SVDB and LTDB external.
Accept only if non-premature S Se improves on >= 2 sources with no accuracy loss on any.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import afe, data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import _rec_id, build_dataset  # noqa: E402
from scripts.dev.exp_pwave import fitp  # noqa: E402
from scripts.final.final_gbt import X_of  # noqa: E402
from scripts.dev.find_nonpremature_s import nonprem_mask  # noqa: E402

A, B = 100, 22                 # window r-A .. r-B (samples @360 Hz)
LSB = 3.0 / 4096 / 300 * 1e3   # ADC LSB, input-referred mV
NP = 10
SOURCES = ("adc", "fine0.05", "fine0.025")


def p_features(sig, r):
    """sig: 1-D P source (mV); r: beat sample indices of one record in order -> (n, NP)."""
    n = len(r); out = np.zeros((n, NP), np.float32)
    T = None; ea = et = en = eg = None
    for i, ri in enumerate(r):
        s, e = int(ri) - A, int(ri) - B
        if s < 0 or e > len(sig):
            continue
        w = sig[s:e].astype(np.float64)
        a0, a1 = w[:6].mean(), w[-6:].mean()
        wd = w - (a0 + (a1 - a0) * np.linspace(0, 1, len(w)))
        pos, neg, tp = wd.max(), -wd.min(), float(np.argmax(wd))
        rough = np.abs(np.diff(wd, 2)).mean()
        if T is None:
            T = wd.copy(); ea, et, en, eg = max(pos, LSB), tp, max(rough, LSB), max(neg, LSB)
        sd = wd.std() * T.std()
        corr = float(((wd - wd.mean()) * (T - T.mean())).mean() / sd) if sd > 0 else 0.0
        dist = float(np.sqrt(((wd - T) ** 2).mean()) / max(np.sqrt((T ** 2).mean()), LSB))
        out[i] = (pos, neg, tp, corr, dist, pos / ea, tp - et, rough / en, neg / eg, (pos - neg) / max(ea + eg, LSB))
        T += (wd - T) / 16; ea += (max(pos, LSB) - ea) / 16; et += (tp - et) / 16
        en += (max(rough, LSB) - en) / 16; eg += (max(neg, LSB) - eg) / 16
    return out


def record_job(args):
    ddir, rec, lead, cfg, r = args
    x, _, _ = data.load_mitdb_record(ddir, rec, lead=lead)
    rng = np.random.default_rng(_rec_id(rec))
    res = {}
    up, dn, y = afe.run_afe(x, cfg, rng)
    res["adc"] = p_features(np.round(y / LSB) * LSB, r)
    for fd in (0.05, 0.025):
        u, d, _ = afe.run_afe(x, replace(cfg, delta_mv=fd), np.random.default_rng(_rec_id(rec)))
        res[f"fine{fd}"] = p_features(np.cumsum(u.astype(np.int64) - d) * fd, r)
    return rec, res


def p_table(ds, recs, ddir, lead, cfg, pool):
    out = {s: np.zeros((len(ds["y"]), NP), np.float32) for s in SOURCES}
    idx = {_rec_id(rec): np.flatnonzero(ds["rec"] == _rec_id(rec)) for rec in recs}
    jobs = [(ddir, rec, lead, cfg, ds["r"][idx[_rec_id(rec)]]) for rec in recs if len(idx[_rec_id(rec)])]
    for rec, res in pool.map(record_job, jobs):
        for s in SOURCES:
            out[s][idx[_rec_id(rec)]] = res[s]
    return out


def main():
    D, DATA, C = sys.argv[1:4]
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, _ = data.get_splits(D)
    names = {k: sorted(p.stem for p in Path(DATA, k).glob("*.hea")) for k in ("incart", "svdb", "ltdb")}
    sets = {"DS1": (ds1, D, "MLII"), "INC": (names["incart"], str(Path(DATA) / "incart"), "II"),
            "SVDB": (names["svdb"], str(Path(DATA) / "svdb"), "ECG1"), "LTDB": (names["ltdb"], str(Path(DATA) / "ltdb"), "ECG1")}
    base = {k: build_dataset(r, p, cfg, cache_dir=C, verbose=False, lead=l) for k, (r, p, l) in sets.items()}
    with ProcessPoolExecutor(2) as pool:
        P = {k: p_table(base[k], r, p, l, cfg, pool) for k, (r, p, l) in sets.items()}
    np.savez_compressed("/tmp/pfeat.npz", **{f"{k}_{s}": P[k][s] for k in P for s in SOURCES})
    yi = base["INC"]["y"].astype(int); keep = np.ones(len(yi), bool)
    keep[np.random.default_rng(0).permutation(np.flatnonzero(yi == 0))[40000:]] = False
    masks = {k: nonprem_mask(base[k]) for k in ("DS1", "SVDB", "LTDB")}

    def show(src, yt, p):
        r = report(yt, p); m = masks[src]
        return (f"{src} acc {100 * r['accuracy']:.2f} S Se {r['S']['se']:.3f} +P {r['S']['ppv']:.3f} F1 {r['S']['f1']:.3f} "
                f"nonprem-S Se {((p == 1) & m).sum() / max(m.sum(), 1):.3f}")

    y, g = base["DS1"]["y"].astype(int), base["DS1"]["rec"]
    for src in (None,) + SOURCES:
        X = {k: X_of(base[k]) if src is None else np.concatenate([X_of(base[k]), P[k][src]], 1) for k in base}
        pd_ = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X["DS1"], y, g):
            pd_[te] = fitp(np.concatenate([X["DS1"][tr], X["INC"][keep]]), np.concatenate([y[tr], yi[keep]])).predict(X["DS1"][te])
        m = fitp(np.concatenate([X["DS1"], X["INC"][keep]]), np.concatenate([y, yi[keep]]))
        line = [show("DS1", y, pd_)] + [show(k, base[k]["y"].astype(int), m.predict(X[k])) for k in ("SVDB", "LTDB")]
        print(f"{str(src or 'base'):9s} | " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
