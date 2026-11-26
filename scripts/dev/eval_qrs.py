"""EC57-style beat detection from comparator events. DS1 only unless --final."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isoline import data, qrs  # noqa: E402
from isoline.afe import run_afe  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", required=True); ap.add_argument("--final", action="store_true")
ap.add_argument("--params", default="")
a = ap.parse_args()
kw = dict(eval(f"dict({a.params})")) if a.params else {}
cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)
ds1, ds2 = data.get_splits(a.data_dir)
recs = ds2 if a.final else ds1
T = [0, 0, 0]; offs = []
for rec in recs:
    x, r, sym = data.load_mitdb_record(a.data_dir, rec)
    up, dn, _ = run_afe(x, cfg, np.random.default_rng(0))
    det = qrs.detect(up, dn, **kw)
    ref = np.asarray(r)
    tp, fn, fp, m = qrs.match(det, ref)
    offs += list(det[m[m >= 0]] - ref[m >= 0])
    T[0] += tp; T[1] += fn; T[2] += fp
    if fn + fp > 0.02 * len(ref):
        print(f"  {rec}: Se {tp / (tp + fn):.4f} +P {tp / max(tp + fp, 1):.4f}  (fn {fn}, fp {fp})")
tp, fn, fp = T
o = np.array(offs)
print(f"{'DS2' if a.final else 'DS1'} {kw}: Se {100 * tp / (tp + fn):.2f} %  +P {100 * tp / (tp + fp):.2f} %  "
      f"(beats {tp + fn}, missed {fn}, false {fp}); fiducial offset median {np.median(o):+.0f} samples, "
      f"|offset| p90 {np.percentile(np.abs(o), 90):.0f}")
