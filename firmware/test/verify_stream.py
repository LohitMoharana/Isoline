"""Streaming firmware path (slot by slot) vs the offline Python reference on detected beats. DS1 records only."""
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
exe = HERE / "_stream"
core = HERE.parent / "core"
subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-o", str(exe), str(HERE / "stream_harness.c"),
                str(core / "isoline_stream.c"), str(core / "isoline_core.c"), "-lm"], check=True)
cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)
T = dict(beats=0, beat_mismatch=0, feat_rows=0, feat_bad=0, af_bad=0)
for rec in recs:
    x, _, _ = data.load_mitdb_record(D, rec)
    up, dn, _ = run_afe(x, cfg, np.random.default_rng(0))
    up = up.astype(np.int16); dn = dn.astype(np.int16)
    det = qrs.detect(up, dn)
    acc = features.leaky_integrator(up, dn)
    py = features.extract_record(up, dn, acc, det, ["N"] * len(det))
    p_feat = np.concatenate([py["hist"].reshape(len(py["r"]), -1), py["res"].reshape(len(py["r"]), -1), py["scal"][:, COLS]], 1)
    p_af = rr_features(det / 360.0)
    with tempfile.TemporaryDirectory() as d:
        inp = Path(d) / "in.bin"
        with open(inp, "wb") as f:
            np.array([len(up), 0], np.int64).tofile(f); up.tofile(f); dn.tofile(f)
        subprocess.run([str(exe), str(inp), str(Path(d) / "o.bin")], check=True)
        raw = np.fromfile(Path(d) / "o.bin", np.uint8).reshape(-1, 24 + 4 * 216 + 40)
    hdr = raw[:, :24].copy().view(np.int64).reshape(-1, 3)
    c_feat = raw[:, 24:24 + 864].copy().view(np.int32)
    c_af = raw[:, 24 + 864:].copy().view(np.float32)
    c_r = hdr[:, 0]
    exp_r = det[1:][det[1:] + 160 <= len(up)]          # every beat after the first whose window has closed
    beats_ok = np.array_equal(c_r, exp_r)
    has = hdr[:, 1] == 1
    rows_ok = np.array_equal(c_r[has], py["r"])
    fbad = int((c_feat[has] != p_feat).any(1).sum()) if rows_ok else len(p_feat)
    idx = np.searchsorted(det, c_r)
    pv = ~np.isnan(p_af[idx]).any(1); cv = hdr[:, 2] == 1
    afbad = int((pv != cv).sum() + (np.abs(c_af[pv & cv] - p_af[idx][pv & cv]) > 0).any(1).sum())
    T["beats"] += len(exp_r); T["beat_mismatch"] += int(not beats_ok); T["feat_rows"] += len(p_feat)
    T["feat_bad"] += fbad; T["af_bad"] += afbad
    print(f"{rec}: beats {'identical' if beats_ok else 'DIFF'} ({len(c_r)}) | features {len(p_feat) - fbad}/{len(p_feat)} | AF mismatches {afbad}", flush=True)
exe.unlink()
print(T)
