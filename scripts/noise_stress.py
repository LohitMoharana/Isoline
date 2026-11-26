"""Noise stress test of the frozen 5.2 KB model (round 7) with MIT-BIH Noise Stress Test DB noise.

Noise (bw = baseline wander, em = electrode motion, ma = muscle artefact; NSTDB channel 1) is added to the
DS2 records at fixed SNRs, EC57-style: S = (median peak-to-peak QRS amplitude)^2 / 8, N = mean square of
the noise segment. Unlike EC57's alternating 2-min segments, noise is applied for the whole record (harsher).
The model is trained once (seed 0) on clean DS1 + INCART and is not changed; nothing is tuned.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ["TINY"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final_gbt as F  # noqa: E402

_clean_loader = data.load_mitdb_record


def make_noisy_loader(noise: np.ndarray, snr_db: float):
    def loader(data_dir, rec, lead="MLII"):
        x, r, sym = _clean_loader(data_dir, rec, lead=lead)
        r = np.asarray(r)
        h = 18                                           # +/- 50 ms at 360 Hz
        p2p = [x[max(i - h, 0):i + h].max() - x[max(i - h, 0):i + h].min() for i in r]
        S = np.median(p2p) ** 2 / 8
        n = noise[: len(x)] - noise[: len(x)].mean()
        scale = np.sqrt(S / (np.mean(n ** 2) * 10 ** (snr_db / 10)))
        return x + scale * n, r, sym
    return loader


def main():
    D, INC, NST, C, OUT = sys.argv[1:6]
    cfg = replace(AFEConfig(), delta_mv=0.1)
    ds1, ds2 = data.get_splits(D)
    d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
    e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
    model = F.fit(F.X_of(d1), d1["y"].astype(int), F.X_of(e), e["y"].astype(int), 0)
    import wfdb
    noises = {k: wfdb.rdrecord(str(Path(NST) / k)).p_signal[:, 0] for k in ("em", "ma", "bw")}
    results = []
    conds = [("clean", None)] + [(k, s) for k in ("em", "ma", "bw") for s in (24, 18, 12, 6, 0)]
    for kind, snr in conds:
        data.load_mitdb_record = _clean_loader if kind == "clean" else make_noisy_loader(noises[kind], snr)
        d2 = build_dataset(ds2, D, cfg, cache_dir=C if kind == "clean" else None, verbose=False)
        r = report(d2["y"].astype(int), model.predict(F.X_of(d2)))
        row = dict(noise=kind, snr_db=snr, accuracy=r["accuracy"], macro_f1=r["macro_f1"],
                   f1_S=r["S"]["f1"], f1_V=r["V"]["f1"], se_V=r["V"]["se"], ppv_V=r["V"]["ppv"],
                   idle_sparsity=d2["stats"]["idle_sparsity"], spikes_per_s=d2["stats"]["spikes_per_s"])
        results.append(row)
        print(f"{kind:5s} {str(snr):>4s} dB  acc {100 * row['accuracy']:6.2f}  macroF1 {row['macro_f1']:.3f}  "
              f"S F1 {row['f1_S']:.3f}  V F1 {row['f1_V']:.3f} (Se {row['se_V']:.3f} +P {row['ppv_V']:.3f})  "
              f"sparsity {100 * row['idle_sparsity']:.1f}%  {row['spikes_per_s']:.0f} spikes/s", flush=True)
        Path(OUT, "noise_stress.json").write_text(json.dumps(results, indent=2))
    data.load_mitdb_record = _clean_loader


if __name__ == "__main__":
    main()
