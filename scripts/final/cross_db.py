"""Cross-database test of the frozen 5.2 KB model (round 7). Trained once (seed 0) on MIT-BIH DS1 + INCART;
tested on databases never used for any design choice in its own selection (SVDB was used as a dev set in
rounds 5 and 8 for other candidates - noted in the README). Nothing is tuned.
For NSRDB (healthy subjects) the key number is the false-alarm rate: non-N predictions per hour.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ["TINY"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report, report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final.final_gbt as F  # noqa: E402

TESTS = [("svdb", "ECG1"), ("edb", "ML"), ("ltdb", "ECG1"), ("nsrdb", "ECG1")]


def main():
    D, INC, ROOTDATA, C, OUT = sys.argv[1:6]
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, _ = data.get_splits(D)
    d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
    model = F.fit(F.X_of(d1), d1["y"].astype(int), F.X_of(e), e["y"].astype(int), 0)
    only = os.environ.get("ONLY")
    outp = Path(OUT, "cross_db.json")
    out = json.loads(outp.read_text()) if (only and outp.exists()) else {}
    for db, lead in TESTS:
        if only and db != only:
            continue
        path = str(Path(ROOTDATA) / db)
        recs = sorted(p.stem for p in Path(path).glob("*.hea"))
        ys, ps, st = [], [], []
        for k in range(0, len(recs), 3):                      # batches keep memory bounded
            d = build_dataset(recs[k:k + 3], path, cfg, cache_dir=C, verbose=False, lead=lead)
            ys.append(d["y"].astype(int)); ps.append(model.predict(F.X_of(d))); st.append(d["stats"]["idle_sparsity"])
            del d
        y, p = np.concatenate(ys), np.concatenate(ps)
        d = {"stats": {"idle_sparsity": float(np.mean(st))}}
        r = report(y, p)
        hours = len(y) / 72 / 60                              # approx. at 72 bpm, for scale only
        fa = int(((p != 0) & (y == 0)).sum())
        r["false_alarms_on_N"] = fa
        r["idle_sparsity"] = d["stats"]["idle_sparsity"]
        out[db] = r
        print(format_report(r, f"== {db} ({len(recs)} records, {len(y)} beats, lead {lead})"))
        print(f"  non-N predictions on N beats: {fa} ({100 * fa / max((y == 0).sum(), 1):.2f}% of N, ~{fa / hours:.0f}/hour)"
              f"  idle sparsity {100 * d['stats']['idle_sparsity']:.1f}%", flush=True)
        Path(OUT, "cross_db.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
