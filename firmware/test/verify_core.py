"""Bit-exactness check of firmware/core against the Python reference on MIT-BIH DS1 records (no DS2)."""
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data, features, qrs  # noqa: E402
from isoline.afe import run_afe  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from scripts.final.af_detector import rr_features  # noqa: E402

COLS = list(range(13)) + [21, 22, 23, 24, 25] + [32, 33, 34, 35, 36, 37]
D = sys.argv[1]
recs = sys.argv[2].split(",") if len(sys.argv) > 2 else data.get_splits(D)[0]
exe = HERE / "_harness"
subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-o", str(exe), str(HERE / "harness.c"), str(HERE.parent / "core" / "isoline_core.c"), "-lm"], check=True)
cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)
tot = dict(acc=0, feat_rows=0, feat_bad=0, qrs=0, qrs_bad=0, af_rows=0, af_maxdiff=0.0, af_valid_mismatch=0)
for rec in recs:
    x, r, sym = data.load_mitdb_record(D, rec)
    up, dn, _ = run_afe(x, cfg, np.random.default_rng(0))
    up = up.astype(np.int16); dn = dn.astype(np.int16); r = np.asarray(r, np.int64)
    acc = features.leaky_integrator(up, dn)
    py = features.extract_record(up, dn, acc, r, sym)
    with tempfile.TemporaryDirectory() as d:
        inp = Path(d) / "in.bin"
        with open(inp, "wb") as f:
            np.array([len(up), len(r)], np.int64).tofile(f); up.tofile(f); dn.tofile(f); r.tofile(f)
        subprocess.run([str(exe), str(inp), str(Path(d) / "o")], check=True)
        c_acc = np.fromfile(Path(d) / "o_acc.bin", np.int16)
        raw = np.fromfile(Path(d) / "o_feat.bin", np.int8).view(np.uint8)
        rows = raw.reshape(-1, 8 + 4 * 216)
        c_idx = rows[:, :8].copy().view(np.int64).ravel(); c_feat = rows[:, 8:].copy().view(np.int32)
        c_qrs = np.fromfile(Path(d) / "o_qrs.bin", np.int64)
        araw = np.fromfile(Path(d) / "o_af.bin", np.int8).view(np.uint8).reshape(-1, 8 + 40)
        a_idx = araw[:, :8].copy().view(np.int64).ravel(); a_feat = araw[:, 8:].copy().view(np.float32)
    p_feat = np.concatenate([py["hist"].reshape(len(py["r"]), -1), py["res"].reshape(len(py["r"]), -1), py["scal"][:, COLS]], 1)
    ok_acc = np.array_equal(c_acc, acc)
    same_rows = len(c_feat) == len(p_feat) and np.array_equal(r[c_idx], py["r"])
    bad = int((c_feat != p_feat).any(1).sum()) if same_rows else len(p_feat)
    p_qrs = qrs.detect(up, dn)
    qbad = 0 if np.array_equal(c_qrs, p_qrs) else max(len(p_qrs), 1)
    p_af = rr_features(r / 360.0)
    pv = ~np.isnan(p_af).any(1)
    cv = a_idx > 0
    ii = np.abs(a_idx)
    both = pv[ii] & cv
    mdiff = float(np.abs(a_feat[both] - p_af[ii][both]).max()) if both.any() else 0.0
    tot["acc"] += int(not ok_acc); tot["feat_rows"] += len(p_feat); tot["feat_bad"] += bad
    tot["qrs"] += len(p_qrs); tot["qrs_bad"] += qbad; tot["af_rows"] += int(both.sum())
    tot["af_maxdiff"] = max(tot["af_maxdiff"], mdiff); tot["af_valid_mismatch"] += int((pv[ii] != cv).sum())
    print(f"{rec}: acc {'OK' if ok_acc else 'DIFF'} | features {len(p_feat) - bad}/{len(p_feat)} identical | "
          f"beats {'identical' if qbad == 0 else 'DIFF'} ({len(p_qrs)}) | AF max |diff| {mdiff:.2e}", flush=True)
print(tot)
