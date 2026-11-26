"""Port of the replicated RF method (scripts/final/replicate_rf.py) onto Isoline's event-driven features.

The 8 features the replica selected on DS1 (R0/avg, avg/R0, R+1/R0, w40_n, R-1/R0, w20_n, w60_n, w40) are
mapped to their comparator-domain equivalents, all computable from spike timing and an up/down counter:
  avg32 RR      -> 64x EWMA of RR (col 21) and pre/that (col 22)
  R+1/R0        -> post_pre_ratio (col 15)   [one-beat decision delay, as in the paper]
  R-1/R0        -> pre/prev_pre (col 8)      [tree models are invariant to the inversion]
  widths        -> staircase widths at 20/40/60 % of peak, raw and normalised (cols 39-41, 43-45)
Classifier: RandomForest, 40 trees, MIT-BIH DS1 only - nothing tuned. Disclosure: the decision to port
this method was made after seeing the replica's DS2 score.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402

import os
COLS = [int(c) for c in os.environ.get("PORT_COLS", "22,21,15,44,8,43,45,40").split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=float(os.environ.get("HP_HZ", "0.5")))
    ds1, ds2 = data.get_splits(a.data_dir)
    d1 = build_dataset(ds1, a.data_dir, cfg, cache_dir=a.cache, verbose=False)
    X1, y1, g1 = d1["scal"][:, COLS].astype(np.float32), d1["y"].astype(int), d1["rec"]
    pred = np.zeros_like(y1)
    for tr, te in GroupKFold(len(np.unique(g1))).split(X1, y1, g1):
        pred[te] = RandomForestClassifier(40, random_state=0, n_jobs=-1).fit(X1[tr], y1[tr]).predict(X1[te])
    r = report(y1, pred)
    print(format_report(r, f"== DS1 leave-one-patient-out (event-domain port, HPF {cfg.hp_hz} Hz, cols {COLS})"), flush=True)
    print(f"DS1 idle sparsity {d1['stats']['idle_sparsity']:.3f}", flush=True)
    out = dict(cols=COLS, ds1_lopo=r)
    if a.final:
        d2 = build_dataset(ds2, a.data_dir, cfg, cache_dir=a.cache, verbose=False)
        X2, y2 = d2["scal"][:, COLS].astype(np.float32), d2["y"].astype(int)
        runs = []
        for s in range(5):
            rr = report(y2, RandomForestClassifier(40, random_state=s, n_jobs=-1).fit(X1, y1).predict(X2))
            runs.append(rr)
            print(format_report(rr, f"== DS2 event-domain port, seed {s}"), flush=True)
        accs = [x["accuracy"] for x in runs]
        print(f"\nDS2 event-domain port: accuracy {100 * np.mean(accs):.2f} +/- {100 * np.std(accs):.2f}, "
              f"idle sparsity {d2['stats']['idle_sparsity']:.3f}")
        out["ds2"] = runs
    (Path(a.out) / "port_rf_events.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
