"""Choose the level-crossing step using DS1 ONLY (record-grouped CV).

Rule fixed in advance: maximise grouped-CV accuracy subject to idle sparsity
> 0.80 measured on DS1. DS2 is never loaded by this script.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose  # noqa: E402
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--out", default=str(ROOT / "results" / "dev" / "select_delta.csv"))
    ap.add_argument("--deltas", type=float, nargs="*", default=[0.05, 0.075, 0.1, 0.15, 0.2])
    a = ap.parse_args()
    ds1, _ = data.get_splits(a.data_dir)          # DS2 deliberately ignored
    rows = []
    for dv in a.deltas:
        ds = build_dataset(ds1, a.data_dir, replace(AFEConfig(), delta_mv=dv),
                           cache_dir=a.cache, verbose=False)
        rep = report(ds["y"], diagnose.cv(ds, False, "sqrt"))
        row = dict(delta_mv=dv, ds1_idle_sparsity=ds["stats"]["idle_sparsity"],
                   cv_accuracy=rep["accuracy"], cv_macro_f1=rep["macro_f1"],
                   cv_f1_S=rep["S"]["f1"], cv_f1_V=rep["V"]["f1"])
        rows.append(row)
        print(row, flush=True)
    ok = [r for r in rows if r["ds1_idle_sparsity"] > 0.80]
    best = max(ok, key=lambda r: r["cv_accuracy"])
    print("SELECTED delta_mv =", best["delta_mv"], best, flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
