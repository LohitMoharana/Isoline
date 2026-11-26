"""Frozen round-8 candidate (chosen by scripts/exp_tiny2.py rule, DS2 not used):
5.2 KB boosted trees (30 rounds, 8 leaves), MIT-BIH DS1 only, HPF 0.05 Hz, round-4 features +
ported RR/QRS-width features + next-RR (one-beat decision delay). Evaluated once on DS2; 5 seeds.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
from scripts.exp_tiny2 import PORT, R4, X_of  # noqa: E402
from scripts.final_gbt import footprint_bytes  # noqa: E402

COLS = R4 + PORT + [15]


def main():
    D, C, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)
    ds1, ds2 = data.get_splits(D)
    d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
    d2 = build_dataset(ds2, D, cfg, cache_dir=C, verbose=False)
    X, y = X_of(d1, COLS), d1["y"].astype(int)
    X2, y2 = X_of(d2, COLS), d2["y"].astype(int)
    print(f"DS2 idle sparsity {d2['stats']['idle_sparsity']:.3f}")
    runs = []
    for s in range(5):
        m = HistGradientBoostingClassifier(max_iter=30, learning_rate=0.2, max_leaf_nodes=8, l2_regularization=1.0,
                                           random_state=s, early_stopping=False)
        m.fit(X, y, sample_weight=np.sqrt(compute_sample_weight("balanced", y)))
        r = report(y2, m.predict(X2)); r["footprint"] = footprint_bytes(m); runs.append(r)
        print(format_report(r, f"== DS2 seed {s} ({r['footprint']['bytes']} B)"), flush=True)
    summ = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in ("accuracy", "macro_f1")}
    for c in "NSVF":
        for mtr in ("f1", "se", "ppv"):
            v = [r[c][mtr] for r in runs]; summ[f"{mtr}_{c}"] = (float(np.mean(v)), float(np.std(v)))
    for k, v in summ.items():
        print(f"  {k:9s} {v[0]:.4f} +/- {v[1]:.4f}")
    Path(OUT, "final_tiny2.json").write_text(json.dumps(dict(cols=COLS, summary=summ, runs=runs,
                                                             ds2_idle_sparsity=d2["stats"]["idle_sparsity"]), indent=2))


if __name__ == "__main__":
    main()
