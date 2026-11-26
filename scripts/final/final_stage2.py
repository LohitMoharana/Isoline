"""Frozen round-5 candidate: two-stage GBT with causal per-patient prediction memory + S-score scale 6.

Chosen without DS2: stage 2 and the S scale maximise the mean accuracy over two development sets
(DS1 grouped CV, and external SVDB with the model trained on DS1 + INCART); see results/dev/exp_s_scale_log.txt
and results/exp_stage2*_log.txt. Ties between scales are broken toward the smaller scale.

    python scripts/final/final_stage2.py --final     # trains on DS1 + INCART, evaluates once on DS2 per seed
"""
from __future__ import annotations

import argparse
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
from scripts.dev.exp_stage2 import X_of, fitp, memory_feats, oof  # noqa: E402

S_SCALE = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--incart-dir", default=str(ROOT / "data" / "incart"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    assert a.final, "this script only exists for the frozen DS2 evaluation; develop with exp_stage2.py"
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, ds2 = data.get_splits(a.data_dir)
    d = build_dataset(ds1, a.data_dir, cfg, cache_dir=a.cache, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(a.incart_dir).glob("*.hea")), a.incart_dir, cfg,
                      cache_dir=a.cache, verbose=False, lead="II")
    t = build_dataset(ds2, a.data_dir, cfg, cache_dir=a.cache, verbose=False)
    Xt, yt = X_of(t), t["y"].astype(int)
    ye = e["y"].astype(int)
    sc = np.array([1, S_SCALE, 1, 1])
    runs = []
    for s in a.seeds:
        rs = np.random.default_rng(s)
        keep = np.ones(len(ye), bool)
        keep[rs.permutation(np.flatnonzero(ye == 0))[40000:]] = False
        X = np.concatenate([X_of(d), X_of(e)[keep]]); y = np.concatenate([d["y"].astype(int), ye[keep]])
        g = np.concatenate([d["rec"], e["rec"][keep]]); c = np.concatenate([d["scal"][:, 38], e["scal"][keep, 38]])
        P = oof(X, y, g)
        m1 = fitp(X, y)
        m2 = fitp(np.concatenate([X, memory_feats(P, g, c)], 1), y, 0.5)
        P1 = m1.predict_proba(Xt).astype(np.float32)
        P2 = m2.predict_proba(np.concatenate([Xt, memory_feats(P1, t["rec"], t["scal"][:, 38])], 1))
        r = report(yt, (P2 * sc).argmax(1))
        r["stage1_unscaled"] = report(yt, P1.argmax(1))
        r["stage2_unscaled"] = report(yt, P2.argmax(1))
        runs.append(r)
        print(format_report(r, f"== DS2 seed {s} (stage 2, S scale {S_SCALE})"), flush=True)
    summ = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
            for k in ("accuracy", "macro_f1")}
    for c in "NSVF":
        for m in ("f1", "se", "ppv"):
            v = [r[c][m] for r in runs]
            summ[f"{m}_{c}"] = (float(np.mean(v)), float(np.std(v)))
    for k in ("stage1_unscaled", "stage2_unscaled"):
        summ[f"{k}_accuracy"] = (float(np.mean([r[k]["accuracy"] for r in runs])),
                                 float(np.std([r[k]["accuracy"] for r in runs])))
    print("\n== summary over seeds", a.seeds)
    for k, v in summ.items():
        print(f"  {k:26s} {v[0]:.4f} +/- {v[1]:.4f}")
    (Path(a.out) / "final_stage2.json").write_text(json.dumps(dict(s_scale=S_SCALE, summary=summ, runs=runs), indent=2))


if __name__ == "__main__":
    main()
