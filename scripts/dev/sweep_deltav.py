"""Accuracy / F1 vs idle sparsity sweep over the level-crossing step (delta V).

Trains on DS1, tests on DS2 (never tuned on DS2). Torch is not required.

    python scripts/dev/sweep_deltav.py --data-dir data/mitdb
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from isoline import baselines, data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.metrics import format_report  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--deltas", type=float, nargs="*",
                    default=[0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2],
                    help="input-referred level-crossing step in mV")
    ap.add_argument("--hp", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--train-jitter", type=int, default=0)
    ap.add_argument("--skip-dense", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds1, ds2 = data.get_splits(a.data_dir)
    print(f"DS1 records: {len(ds1)}   DS2 records: {len(ds2)}")
    base = AFEConfig(hp_hz=a.hp)
    rows = []

    def run(label, mode, cfg, dense):
        tr = build_dataset(ds1, a.data_dir, cfg, jitter=a.train_jitter, want_dense=dense,
                           cache_dir=a.cache, verbose=False)
        te = build_dataset(ds2, a.data_dir, cfg, want_dense=dense,
                           cache_dir=a.cache, verbose=False)
        rep = baselines.fit_eval(tr, te, mode, max_iter=a.max_iter)
        print(format_report(rep, f"== {label}  sparsity={rep['idle_sparsity']:.3f}"
                                 f"  spikes/s={rep['spikes_per_s']:.1f}"))
        rows.append(dict(
            label=label, mode=mode, delta_mv=cfg.delta_mv if mode == "spike" else "",
            idle_sparsity=rep["idle_sparsity"] if mode == "spike" else "",
            spikes_per_s=rep["spikes_per_s"] if mode == "spike" else "",
            accuracy=rep["accuracy"], macro_f1=rep["macro_f1"],
            f1_N=rep["N"]["f1"], f1_S=rep["S"]["f1"], f1_V=rep["V"]["f1"], f1_F=rep["F"]["f1"]))

    if not a.skip_dense:
        run("dense_raw (no AFE)", "dense_raw", base, True)
        run("dense_afe (post-HPF/LPF)", "dense_afe", base, True)
    for dv in a.deltas:
        run(f"spike dV={dv} mV", "spike", replace(base, delta_mv=dv), False)

    csv_path = out / "sweep_deltav.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sp = [r for r in rows if r["mode"] == "spike"]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = [r["idle_sparsity"] * 100 for r in sp]
        ax.plot(xs, [r["accuracy"] * 100 for r in sp], "o-", label="accuracy (%)")
        ax.plot(xs, [r["f1_V"] * 100 for r in sp], "s-", label="V F1 (x100)")
        ax.plot(xs, [r["f1_S"] * 100 for r in sp], "^-", label="S F1 (x100)")
        ax.axvline(80, ls="--", c="gray", lw=1)
        ax.set_xlabel("idle sparsity (% of 360 Hz slots with no spike)")
        ax.set_ylabel("DS2 score")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "sweep_deltav.png", dpi=150)
        print(f"wrote {out / 'sweep_deltav.png'}")
    except Exception as exc:  # plotting is optional
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
