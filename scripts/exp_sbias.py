"""Option 1: S-biased operating point (dev only; DS2 never loaded).

Rule fixed before any result: final class = argmax(p * [1, b, 1, 1]); b is chosen to maximise the macro F1 (N,S,V,F)
of DS1 record-grouped 5-fold out-of-fold predictions, with the RR-only AF detector retrained per fold on AFDB + the
training-fold DS1 records and applied (S -> N during AF) exactly as in deployment. Config: deployed (HPF 0.05 Hz).
Done for the full (150 x 31) and tiny (30 x 8) models, seed 0.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import _rec_id, build_dataset  # noqa: E402
import scripts.final_gbt as F  # noqa: E402
from scripts.af_detector import afdb_records, mit_records, p_af, rr_features, train  # noqa: E402

GRID = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 14, 20, 28, 40, 56, 80]   # extended after b=20 hit the edge (DS1 only)


def main():
    D, INC, AF, C, OUT = sys.argv[1:6]
    cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)
    ds1, _ = data.get_splits(D)
    d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
    X1, y1, g1 = F.X_of(d1), d1["y"].astype(int), d1["rec"]
    Xe, ye = F.X_of(e), e["y"].astype(int)
    Xa, ya, _ = afdb_records(AF)
    Fr = np.zeros((len(y1), Xa.shape[1]), np.float32)
    for rec in np.unique(g1):
        k = np.flatnonzero(g1 == rec); Fr[k] = rr_features(d1["r"][k] / 360)
    id2name = {_rec_id(r): r for r in ds1}
    folds = list(GroupKFold(5).split(X1, y1, g1))
    afm = np.zeros(len(y1), bool)
    for tr, te in folds:
        tr_recs = sorted({id2name[int(x)] for x in np.unique(g1[tr])})
        Xm, ym, _ = mit_records(D, tr_recs)
        afm[te] = p_af(train(Xa, ya, Xm, ym), Fr[te]) > 0.5
    res = {}
    for size, tiny in (("full", False), ("tiny", True)):
        F.TINY = tiny
        cache = Path(f"/tmp/oof_{size}.npy")
        if cache.exists():
            P = np.load(cache)
        else:
            P = np.zeros((len(y1), 4))
            for tr, te in folds:
                P[te] = F.fit(X1[tr], y1[tr], Xe, ye, 0).predict_proba(X1[te])
            np.save(cache, P)
        rows = []
        for b in GRID:
            p = np.argmax(P * np.array([1, b, 1, 1]), 1); p[(p == 1) & afm] = 0
            r = report(y1, p)
            rows.append(dict(b=b, acc=r["accuracy"], macro_f1=r["macro_f1"], S_se=r["S"]["se"], S_ppv=r["S"]["ppv"],
                             S_f1=r["S"]["f1"], V_f1=r["V"]["f1"]))
            print(f"{size:4s} b={b:5.2f}  acc {100 * r['accuracy']:.2f}  macroF1 {r['macro_f1']:.3f}  S Se {r['S']['se']:.3f} "
                  f"+P {r['S']['ppv']:.3f} F1 {r['S']['f1']:.3f}  V F1 {r['V']['f1']:.3f}", flush=True)
        best = max(rows, key=lambda z: z["macro_f1"])
        print(f"{size}: chosen b = {best['b']} (max DS1 OOF macro F1)", flush=True)
        res[size] = dict(grid=rows, chosen_b=best["b"])
    Path(OUT, "exp_sbias.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
