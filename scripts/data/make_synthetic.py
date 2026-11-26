"""Generate a tiny SYNTHETIC MIT-BIH-shaped dataset for smoke tests only.

The numbers obtained from this data are meaningless for research: it exists
purely to exercise the loader / AFE / feature / training code paths when the
real PhysioNet data is not available.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isoline.config import DS1, DS2  # noqa: E402

FS = 360


def g(t, t0, w, a):
    return a * np.exp(-0.5 * ((t - t0) / w) ** 2)


def beat_template(kind: str, t: np.ndarray, rng) -> np.ndarray:
    """t in seconds relative to the R peak."""
    if kind == "N" or kind == "S":
        y = (g(t, -0.18, 0.025, 0.12 if kind == "N" else 0.05) + g(t, -0.025, 0.010, -0.10)
             + g(t, 0.0, 0.012, 1.1) + g(t, 0.025, 0.010, -0.25) + g(t, 0.25, 0.05, 0.30))
    elif kind == "V":
        s = rng.choice([-1.0, 1.0])
        y = s * g(t, 0.0, 0.040, 1.5) - s * g(t, 0.28, 0.07, 0.45)
    else:  # F: fusion
        y = 0.5 * (g(t, 0.0, 0.012, 1.1) + g(t, 0.0, 0.035, 1.2)) + g(t, 0.27, 0.06, 0.2)
    return y


def synth_record(rng, minutes: float = 4.0):
    n = int(minutes * 60 * FS)
    sig = np.zeros(n)
    mean_rr = rng.uniform(0.7, 1.0) * FS
    t, prev_ectopic = 0.6 * FS, False
    samples, symbols = [], []
    while t < n - 0.6 * FS:
        u = rng.random()
        if prev_ectopic:
            kind, rr, prev_ectopic = "N", mean_rr * rng.uniform(1.25, 1.5), False
        elif u < 0.08:
            kind, rr, prev_ectopic = "V", mean_rr * rng.uniform(0.55, 0.75), True
        elif u < 0.15:
            kind, rr, prev_ectopic = "S", mean_rr * rng.uniform(0.6, 0.8), False
        elif u < 0.17:
            kind, rr = "F", mean_rr * rng.normal(1, 0.04)
        else:
            kind, rr = "N", mean_rr * rng.normal(1, 0.04)
        t += rr
        lo, hi = int(t - 0.5 * FS), int(t + 0.5 * FS)
        if lo < 0 or hi >= n:
            break
        tt = (np.arange(lo, hi) - t) / FS
        sig[lo:hi] += beat_template(kind, tt, rng)
        samples.append(int(round(t)))
        symbols.append(kind if kind != "S" else "A")
    tt = np.arange(n) / FS
    sig += 0.25 * np.sin(2 * np.pi * 0.2 * tt + rng.uniform(0, 6)) + rng.normal(0, 0.02, n)
    return sig, np.array(samples), symbols


def main() -> None:
    import wfdb
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "synthetic_mitdb"))
    ap.add_argument("--per-split", type=int, default=5)
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for rec in DS1[: a.per_split] + DS2[: a.per_split]:
        sig, samples, symbols = synth_record(rng, a.minutes)
        p = np.stack([sig, 0.6 * sig + rng.normal(0, 0.02, len(sig))], axis=1)
        wfdb.wrsamp(rec, fs=FS, units=["mV", "mV"], sig_name=["MLII", "V1"],
                    p_signal=p, fmt=["212", "212"], adc_gain=[200.0, 200.0],
                    baseline=[1024, 1024], write_dir=str(out))
        wfdb.wrann(rec, "atr", sample=samples, symbol=symbols, write_dir=str(out))
    print(f"wrote synthetic records to {out}")


if __name__ == "__main__":
    main()
