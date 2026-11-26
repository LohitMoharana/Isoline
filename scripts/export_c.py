"""Export the round-10 tiny system (5.2 KB beat classifier + RR-only AF detector) to C and verify it.

* Beat-classifier inputs are integers (spike counts, template residuals, integer scalars), so every split
  `x <= t` is exported as the exact integer test `x <= floor(t)`.
* Leaves are exported as float32 (exact reference) and as int8 with one scale per model (deployment).
* The AF detector uses RR ratios (floats); it is exported in float32 (Cortex-M4F has an FPU).
The test compiles the C with gcc and compares its decisions with scikit-learn on every DS1 and INCART beat
(DS2 is not used). Output: export/isoline_model.h, export/isoline_model.c, export/test_harness.c, results/export_c_log.txt
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ["TINY"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final_gbt as F  # noqa: E402
from scripts.af_detector import afdb_records, mit_records, rr_features, train  # noqa: E402


def flatten(clf):
    """Return per-tree node tables and the baseline (raw score offset) of a HistGradientBoostingClassifier."""
    trees = []
    for it in clf._predictors:
        for k, tp in enumerate(it):
            trees.append((k, tp.nodes))
    base = np.asarray(clf._baseline_prediction).reshape(-1)
    return trees, base


def raw_python(trees, base, X, integer):
    ncls = len(base)
    raw = np.tile(base, (len(X), 1)).astype(np.float64)
    for k, nd in trees:
        for i in range(len(X)):
            j = 0
            while not nd["is_leaf"][j]:
                f = nd["feature_idx"][j]
                t = np.floor(nd["num_threshold"][j]) if integer else nd["num_threshold"][j]
                j = nd["left"][j] if X[i, f] <= t else nd["right"][j]
            raw[i, k % ncls] += nd["value"][j]
    return raw


def emit(name, trees, base, integer, n_features):
    feat, thr, left, right, leaf, start = [], [], [], [], [], []
    for _, nd in trees:
        start.append(len(feat))
        for j in range(len(nd)):
            isl = bool(nd["is_leaf"][j])
            feat.append(255 if isl else int(nd["feature_idx"][j]))
            thr.append(0 if isl else (int(np.floor(nd["num_threshold"][j])) if integer else float(nd["num_threshold"][j])))
            left.append(int(nd["left"][j]) if not isl else 0)
            right.append(int(nd["right"][j]) if not isl else 0)
            leaf.append(float(nd["value"][j]) if isl else 0.0)
    leaf = np.array(leaf, np.float32)
    scale = float(np.abs(leaf).max() / 127) if np.abs(leaf).max() > 0 else 1.0
    q = np.clip(np.round(leaf / scale), -127, 127).astype(np.int8)
    ttype = "int32_t" if integer else "float"
    fmt = (lambda v: str(v)) if integer else (lambda v: f"{v:.9e}f")
    cls_of_tree = [k % len(base) for k, _ in trees]
    J = lambda xs: ",".join(xs)
    c = f"""
/* {name}: {len(trees)} trees, {len(feat)} nodes, {n_features} input features */
static const uint8_t  {name}_feat[{len(feat)}] = {{{J(str(v) for v in feat)}}};
static const {ttype} {name}_thr[{len(feat)}] = {{{J(fmt(v) for v in thr)}}};
static const uint16_t {name}_left[{len(feat)}] = {{{J(str(v) for v in left)}}};
static const uint16_t {name}_right[{len(feat)}] = {{{J(str(v) for v in right)}}};
static const float    {name}_leaf_f32[{len(feat)}] = {{{J(f'{v:.9e}f' for v in leaf)}}};
static const int8_t   {name}_leaf_q[{len(feat)}] = {{{J(str(int(v)) for v in q)}}};
static const float    {name}_leaf_scale = {scale:.9e}f;
static const uint16_t {name}_start[{len(trees)}] = {{{J(str(v) for v in start)}}};
static const uint8_t  {name}_cls[{len(trees)}] = {{{J(str(v) for v in cls_of_tree)}}};
static const float    {name}_base[{len(base)}] = {{{J(f'{v:.9e}f' for v in base)}}};
#define {name.upper()}_NTREES {len(trees)}
#define {name.upper()}_NCLS {len(base)}
"""
    size = dict(nodes=len(feat), internal=int(sum(1 for f in feat if f != 255)), leaves=int(sum(1 for f in feat if f == 255)))
    return c, size


C_FUNCS = r"""
/* Traverse one model. use_q: 1 = int8 leaves (deployment), 0 = float32 leaves (reference). */
#define DEFINE_PREDICT(NAME, XTYPE, TTYPE)                                                        \
static void NAME##_raw(const XTYPE *x, int use_q, float *raw) {                                   \
    for (int k = 0; k < NAME##_NCLS_M; k++) raw[k] = NAME##_base[k];                               \
    for (int t = 0; t < NAME##_NTREES_M; t++) {                                                    \
        int j = NAME##_start[t];                                                                   \
        while (NAME##_feat[j] != 255) {                                                            \
            j = NAME##_start[t] + ((x[NAME##_feat[j]] <= (TTYPE)NAME##_thr[j]) ? NAME##_left[j] : NAME##_right[j]); \
        }                                                                                          \
        raw[NAME##_cls[t]] += use_q ? NAME##_leaf_q[j] * NAME##_leaf_scale : NAME##_leaf_f32[j];  \
    }                                                                                              \
}
#define beat_NCLS_M BEAT_NCLS
#define beat_NTREES_M BEAT_NTREES
#define af_NCLS_M AF_NCLS
#define af_NTREES_M AF_NTREES
DEFINE_PREDICT(beat, int32_t, int32_t)
DEFINE_PREDICT(af, float, float)

/* Returns 0=N 1=S 2=V 3=F. af_feat may be NULL when fewer than 8 previous RR intervals exist. */
int isoline_classify(const int32_t *beat_feat, const float *af_feat, int use_q) {
    float raw[BEAT_NCLS];
    beat_raw(beat_feat, use_q, raw);
    int best = 0;
    for (int k = 1; k < BEAT_NCLS; k++) if (raw[k] > raw[best]) best = k;
    if (best == 1 && af_feat) {                       /* AF flag: do not report S during AF/AFL */
        float a[1]; af_raw(af_feat, use_q, a);
        if (a[0] > 0.0f) best = 0;                    /* logit > 0  <=>  P(AF) > 0.5 */
    }
    return best;
}
"""

HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "isoline_model.h"
int main(int argc, char **argv) {
    FILE *fb = fopen(argv[1], "rb"), *fa = fopen(argv[2], "rb"), *fo = fopen(argv[3], "w");
    int n = atoi(argv[4]), nb = atoi(argv[5]), na = atoi(argv[6]), use_q = atoi(argv[7]);
    int32_t *xb = malloc(sizeof(int32_t) * nb); float *xa = malloc(sizeof(float) * (na + 1));
    for (int i = 0; i < n; i++) {
        if (fread(xb, sizeof(int32_t), nb, fb) != (size_t)nb) return 1;
        if (fread(xa, sizeof(float), na + 1, fa) != (size_t)(na + 1)) return 1;  /* last value: valid flag */
        fprintf(fo, "%d\n", isoline_classify(xb, xa[na] > 0.5f ? xa : NULL, use_q));
    }
    fclose(fo); return 0;
}
"""


def main():
    D, INC, AF, C = sys.argv[1:5]
    out = ROOT / "export"; out.mkdir(exist_ok=True)
    cfg = replace(AFEConfig(), delta_mv=0.1, hp_hz=0.05)   # the circuit's 0.05 Hz high-pass
    ds1, _ = data.get_splits(D)
    d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
    X1, y1, Xe, ye = F.X_of(d1), d1["y"].astype(int), F.X_of(e), e["y"].astype(int)
    beat = F.fit(X1, y1, Xe, ye, 0)
    Xa, ya, _ = afdb_records(AF); Xm, ym, _ = mit_records(D, ds1)
    det = train(Xa, ya, Xm, ym)
    # verification rows: DS1 + INCART (never DS2)
    Xv = np.concatenate([X1, Xe]); recv = np.concatenate([d1["rec"], e["rec"]]); rv = np.concatenate([d1["r"], e["r"]])
    assert np.all(Xv == np.round(Xv)), "beat features must be integers"
    Av = np.zeros((len(Xv), Xa.shape[1]), np.float32)
    for g in np.unique(recv):
        k = np.flatnonzero(recv == g); Av[k] = rr_features(rv[k] / 360)
    valid = ~np.isnan(Av).any(1)
    # reference decision in Python (sklearn)
    pb = beat.predict(Xv)
    paf = np.zeros(len(Xv), bool); paf[valid] = det.predict_proba(Av[valid])[:, 1] > 0.5
    ref = pb.copy(); ref[(ref == 1) & paf] = 0
    # emit C
    tb, bb = flatten(beat); ta, ba = flatten(det)
    cb, sb = emit("beat", tb, bb, True, Xv.shape[1]); ca, sa = emit("af", ta, ba, False, Xa.shape[1])
    (out / "isoline_model.h").write_text("#pragma once\n#include <stdint.h>\n" + cb + ca + C_FUNCS)
    (out / "test_harness.c").write_text(HARNESS)
    Xv.astype(np.int32).tofile(out / "_beat.bin")
    np.concatenate([np.nan_to_num(Av, nan=0.0), valid[:, None].astype(np.float32)], 1).astype(np.float32).tofile(out / "_af.bin")
    exe = out / "_harness"
    subprocess.run(["gcc", "-O2", "-std=c99", "-o", str(exe), str(out / "test_harness.c"), "-lm"], check=True, cwd=out)
    lines = []
    for use_q in (0, 1):
        subprocess.run([str(exe), str(out / "_beat.bin"), str(out / "_af.bin"), str(out / "_pred.txt"), str(len(Xv)),
                        str(Xv.shape[1]), str(Xa.shape[1]), str(use_q)], check=True)
        pc = np.loadtxt(out / "_pred.txt", dtype=int)
        agree = float(np.mean(pc == ref))
        lines.append(f"{'int8 leaves' if use_q else 'float32 leaves'}: C vs Python agreement on {len(Xv)} beats = "
                     f"{100 * agree:.4f}% ({int((pc != ref).sum())} differ)")
    size_q = lambda s, tb_: s["internal"] * (1 + (4 if tb_ else 4) + 2 + 2) + s["leaves"] * 1
    lines.append(f"beat model: {sb['nodes']} nodes ({sb['internal']} splits, {sb['leaves']} leaves); "
                 f"AF detector: {sa['nodes']} nodes ({sa['internal']} splits, {sa['leaves']} leaves)")
    lines.append(f"as exported (uint8 feat, 32-bit thr, 2x uint16 child, int8 leaf): beat {size_q(sb, True)} B, "
                 f"AF {size_q(sa, False)} B; with 16-bit thresholds and implicit left child the beat model is "
                 f"{sb['internal'] * 5 + sb['leaves']} B")
    for f in ("_beat.bin", "_af.bin", "_pred.txt", "_harness"):
        (out / f).unlink(missing_ok=True)
    txt = "\n".join(lines); print(txt)
    (ROOT / "results" / "export_c_log.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
