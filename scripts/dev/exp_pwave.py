"""P-wave detail from a second, finer level-crossing band, read only in the pre-QRS (P-wave) bins. Dev only:
DS1 grouped 5-fold CV and external SVDB (trained on DS1 + INCART). DS2 never loaded.

Extra features per beat (all causal): fine-band up/down histograms and template residual in pre-R bins 0..11,
P-region spike count (bins 2-9), its ratio to the patient's running average (EWMA 1/16), and the P-region
residual L1 distance and its ratio to the running average.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
from scripts.final.final_gbt import X_of  # noqa: E402


def p_feats(f, rec):
    n = len(f["y"])
    h = f["hist"][:, :2, :12].reshape(n, -1).astype(np.float32)
    rs = f["res"][:, :2, :12].reshape(n, -1).astype(np.float32)
    pc = f["hist"][:, :2, 2:10].sum((1, 2)).astype(np.float32)
    pr = np.abs(f["res"][:, :2, 2:10]).sum((1, 2)).astype(np.float32)
    ratio_c, ratio_r = np.ones(n, np.float32), np.ones(n, np.float32)
    for g in np.unique(rec):
        k = np.flatnonzero(rec == g); ec = er = None
        for i in k:
            if ec is None:
                ec, er = max(pc[i], 1), max(pr[i], 1)
            ratio_c[i], ratio_r[i] = pc[i] / max(ec, 1), pr[i] / max(er, 1)
            ec += (pc[i] - ec) / 16; er += (pr[i] - er) / 16
    return np.concatenate([h, rs, np.stack([pc, ratio_c, pr, ratio_r], 1)], 1)


def gbt():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)


def fitp(X, y):
    return gbt().fit(X, y, sample_weight=np.sqrt(compute_sample_weight("balanced", y)))


def main():
    D, INC, SV, C = sys.argv[1:5]
    fines = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.0, 0.05, 0.025]
    ds1, _ = data.get_splits(D)
    sets = {"mit": (ds1, D, "MLII"), "inc": (sorted(p.stem for p in Path(INC).glob("*.hea")), INC, "II"),
            "sv": (sorted(p.stem for p in Path(SV).glob("*.hea")), SV, "ECG1")}
    base = {k: build_dataset(r, p, replace(AFEConfig(), delta_mv=0.1), cache_dir=C, verbose=False, lead=l) for k, (r, p, l) in sets.items()}
    keep = np.ones(len(base["inc"]["y"]), bool)
    yi = base["inc"]["y"].astype(int)
    keep[np.random.default_rng(0).permutation(np.flatnonzero(yi == 0))[40000:]] = False
    for fd in fines:
        X = {}
        for k, (r, p, l) in sets.items():
            xb = X_of(base[k])
            if fd:
                f = build_dataset(r, p, replace(AFEConfig(), delta_mv=fd), cache_dir=C, verbose=False, lead=l)
                assert (f["y"] == base[k]["y"]).all()
                xb = np.concatenate([xb, p_feats(f, base[k]["rec"])], 1)
                sp = f["stats"]["idle_sparsity"]
            else:
                sp = base[k]["stats"]["idle_sparsity"]
            X[k] = (xb, sp)
        y, g = base["mit"]["y"].astype(int), base["mit"]["rec"]
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X["mit"][0], y, g):
            pred[te] = fitp(np.concatenate([X["mit"][0][tr], X["inc"][0][keep]]), np.concatenate([y[tr], yi[keep]])).predict(X["mit"][0][te])
        a = report(y, pred)
        m = fitp(np.concatenate([X["mit"][0], X["inc"][0][keep]]), np.concatenate([y, yi[keep]]))
        b = report(base["sv"]["y"].astype(int), m.predict(X["sv"][0]))
        print(f"fine dV {fd:5.3f} (fine-band sparsity on DS1 {100 * X['mit'][1]:.1f}%) | DS1 5-fold acc {100 * a['accuracy']:.2f} "
              f"S F1 {a['S']['f1']:.3f} | SVDB acc {100 * b['accuracy']:.2f} S Se {b['S']['se']:.3f} +P {b['S']['ppv']:.3f} "
              f"F1 {b['S']['f1']:.3f} V F1 {b['V']['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
