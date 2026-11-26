"""Virtual HIL scoring: MIT-BIH record -> ngspice-derived circuit response -> comparator/DAC/firmware simulator
(hil_sim, running the unmodified STM32 application) -> UART beat list -> scored against the reference annotations
with the same EC57 +/-150 ms matching and AAMI mapping as scripts/final/eval_end_to_end.py.

usage: python3 hil_run.py <mitdb dir> <DS1|DS2|rec,rec,...> <out.json> [H_mV] [off_up_mV] [off_dn_mV] [VDDA]
The model inside the firmware is export/isoline_model.h (tiny GBT, one seed, trained on DS1 + INCART only).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "hardware" / "sim")); sys.path.insert(0, str(ROOT / "hardware"))
from isoline import data, qrs  # noqa: E402
from isoline.config import AAMI_MAP  # noqa: E402
from spice_frontend import spice_afe_fast  # noqa: E402
import design as DZ  # noqa: E402

CLS = "NSVF"
TSLOT = 91 / 32768


def build(exe):
    src = HERE.parent
    import os
    cmd = ["gcc", "-O2", "-DISO_HOST_SIM", *os.environ.get("HIL_CFLAGS", "").split(), f"-I{HERE}", f"-I{src.parent / 'core'}", f"-I{ROOT / 'export'}",
           str(HERE / "hil_sim.c"), str(src.parent / "core" / "isoline_core.c"), str(src.parent / "core" / "isoline_stream.c"),
           "-lm", "-o", str(exe)]
    subprocess.run(cmd, check=True)


def run_record(exe, D, rec, args):
    x, ref, sym = data.load_mitdb_record(D, rec)
    _, _, y_mv, clip = spice_afe_fast(x, 0.1)
    vo = (DZ.VREF + y_mv * 1e-3 * DZ.TOTAL_GAIN).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".f32") as f:
        vo.tofile(f.name)
        out = subprocess.run([str(exe), f.name, *args], capture_output=True, text=True, check=True).stdout
    det, lab = [], []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 2 and p[1] in CLS:
            det.append(int(round(int(p[0]) * TSLOT * 360))); lab.append(CLS.index(p[1]))   # slot -> 360 Hz sample
    return np.array(det, int), np.array(lab, int), np.asarray(ref), sym, clip


def main():
    D, which, outp = sys.argv[1:4]
    args = sys.argv[4:]
    ds1, ds2 = data.get_splits(D)
    recs = ds1 if which == "DS1" else ds2 if which == "DS2" else which.split(",")
    exe = Path(tempfile.mkdtemp()) / "hil_sim_bin"; build(exe)
    conf = np.zeros((4, 4), int); tp = fn = fp = 0; missed = 0; per = {}
    for rec in recs:
        det, lab, ref, sym, clip = run_record(exe, D, rec, args)
        t, n, p, m = qrs.match(det, ref)
        tp += t; fn += n; fp += p
        order = np.argsort(det); det_s, lab_s = det[order], lab[order]
        c = np.zeros((4, 4), int); ms = 0
        for i, s in enumerate(sym):
            a = AAMI_MAP.get(s)
            if a not in CLS:
                continue
            if m[i] < 0:
                ms += 1; continue
            c[CLS.index(a), lab_s[m[i]]] += 1
        conf += c; missed += ms
        per[rec] = {"acc": float(np.trace(c) / max(c.sum(), 1)), "Se": t / max(t + n, 1), "PPV": t / max(t + p, 1),
                    "beats": int(c.sum()), "missed": ms, "clip": clip}
        print(f"{rec}: matched {c.sum():5d}  acc {100 * per[rec]['acc']:6.2f}%  Se {100 * per[rec]['Se']:6.2f}%  "
              f"+P {100 * per[rec]['PPV']:6.2f}%", flush=True)
    acc = np.trace(conf) / conf.sum()
    rep = {"records": which, "args": args, "confusion_rows_true_NSVF": conf.tolist(), "matched_accuracy": float(acc),
           "end_to_end_accuracy": float(np.trace(conf) / (conf.sum() + missed)),
           "detection_Se": tp / (tp + fn), "detection_PPV": tp / (tp + fp), "per_record": per}
    for k, cl in enumerate(CLS):
        se = conf[k, k] / max(conf[k].sum(), 1); pp = conf[k, k] / max(conf[:, k].sum(), 1)
        rep[f"{cl}_Se"], rep[f"{cl}_PPV"] = float(se), float(pp)
    print(f"== {which} {args}: matched accuracy {100 * acc:.2f}%  end-to-end {100 * rep['end_to_end_accuracy']:.2f}%  "
          f"detection Se {100 * rep['detection_Se']:.2f}% +P {100 * rep['detection_PPV']:.2f}%")
    print("   " + "  ".join(f"{c}: Se {100 * rep[c + '_Se']:.1f} +P {100 * rep[c + '_PPV']:.1f}" for c in CLS))
    print("confusion (rows true N,S,V,F):\n" + str(conf))
    Path(outp).write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
