"""AF flag on the round-4 model (and the tiny model) with T chosen by DS1 LOPO beat accuracy (rule fixed here),
then one DS2 evaluation over 5 seeds."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final_gbt as F  # noqa: E402
from scripts.alarms import T_GRID, apply_af, irregularity  # noqa: E402


def main():
    D, INC, C, OUT = sys.argv[1:5]
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, ds2 = data.get_splits(D)
    d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
    d2 = build_dataset(ds2, D, cfg, cache_dir=C, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
    X1, y1, g1 = F.X_of(d1), d1["y"].astype(int), d1["rec"]
    Xe, ye = F.X_of(e), e["y"].astype(int)
    irr1, irr2 = irregularity(d1["r"], g1), irregularity(d2["r"], d2["rec"])
    p1 = np.zeros_like(y1)
    for tr, te in GroupKFold(len(np.unique(g1))).split(X1, y1, g1):
        p1[te] = F.fit(X1[tr], y1[tr], Xe, ye, 0).predict(X1[te])
    accs = {str(T): report(y1, apply_af(p1, irr1, T))["accuracy"] for T in T_GRID}
    T = max(T_GRID, key=lambda t: (accs[str(t)], t is None))
    print("DS1 LOPO accuracy by T:", {k: round(100 * v, 2) for k, v in accs.items()}, "-> T =", T, flush=True)
    X2, y2 = F.X_of(d2), d2["y"].astype(int)
    runs = []
    for s in range(5):
        p = F.fit(X1, y1, Xe, ye, s).predict(X2)
        r = report(y2, apply_af(p, irr2, T)); r["no_flag_accuracy"] = report(y2, p)["accuracy"]; runs.append(r)
        print(format_report(r, f"== DS2 seed {s}, AF flag T={T} (no flag {100 * r['no_flag_accuracy']:.2f}%)"), flush=True)
    summ = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in ("accuracy", "macro_f1", "no_flag_accuracy")}
    for c in "NSVF":
        for m in ("f1", "se", "ppv"):
            v = [r[c][m] for r in runs]; summ[f"{m}_{c}"] = (float(np.mean(v)), float(np.std(v)))
    for k, v in summ.items():
        print(f"  {k:16s} {v[0]:.4f} +/- {v[1]:.4f}")
    tag = "tiny" if F.TINY else "round4"
    Path(OUT, f"af_{tag}.json").write_text(json.dumps(dict(T=T, ds1_lopo_by_T=accs, summary=summ, runs=runs), indent=2, default=float))


if __name__ == "__main__":
    main()
