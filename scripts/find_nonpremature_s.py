"""Count non-premature S beats (pre-RR >= 0.9 x dominant RR, the 75th percentile of the previous 16 RR) in the
databases we may use for development. DS2 is not loaded."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402


def nonprem_mask(d):
    m = np.zeros(len(d["y"]), bool)
    for g in np.unique(d["rec"]):
        k = np.flatnonzero(d["rec"] == g); t = d["r"][k] / 360; rr = np.diff(t, prepend=t[0])
        for j, i in enumerate(k):
            if j > 8 and d["y"][i] == 1:
                m[i] = rr[j] >= 0.9 * np.percentile(rr[max(j - 16, 1):j], 75)
    return m


if __name__ == "__main__":
    D, DATA, C = sys.argv[1:4]
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, _ = data.get_splits(D)
    sets = [("MIT DS1", D, ds1, "MLII")] + [(db.upper(), str(Path(DATA) / db), sorted(p.stem for p in Path(DATA, db).glob("*.hea")), lead)
                                            for db, lead in (("incart", "II"), ("svdb", "ECG1"), ("ltdb", "ECG1"))]
    for name, path, recs, lead in sets:
        d = build_dataset(recs, path, cfg, cache_dir=C, verbose=False, lead=lead)
        m = nonprem_mask(d); s = d["y"] == 1
        per = {str(g): int((m & (d["rec"] == g)).sum()) for g in np.unique(d["rec"][m])}
        top = sorted(per.items(), key=lambda x: -x[1])[:6]
        print(f"{name:8s} S beats {s.sum():6d}  non-premature {m.sum():5d} ({100 * m.sum() / max(s.sum(), 1):.1f}%)  "
              f"in {len(per)} records; top {top}", flush=True)
