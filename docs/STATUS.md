# Status & hand-off notes

_Last updated: 2026-11-24. Read this first when resuming._

## Where we are

The Python proof-of-concept (Phase 1–2) is **complete enough to evaluate**, but it has **not
cleared the accuracy gates**. Hardware (compute platform, PCB, RTL) is deliberately **undecided**
until the PoC is finished.

| Gate | Target | Current (DS2, int8 student, 5 seeds) |
|---|---|---|
| Idle sparsity | > 80 % | **89.7 %** ✅ |
| Model footprint | ≤ 12 KB | **9,072 B** ✅ |
| Overall accuracy | ≥ 95.5 % | 93.7 ± 0.4 % ❌ |
| V F1 | ≥ 0.91 | 0.891 ± 0.014 ❌ |
| S F1 / F F1 | (no gate yet) | 0.167 / 0.002 — weak |

Context for the accuracy number: always predicting "normal" scores 89.0 % on DS2, so 93.7 % is only
4.7 points above trivial. Per-class F1 is the honest metric.

## Rules we agreed to follow (do not break these)

1. **DS2 is the test set. Never tune on it.** All choices (dV, features, loss, early stopping,
   hyper-parameters) use DS1 only (record-grouped CV or the `DS1_VAL` hold-out).
   `scripts/train_student.py` only loads DS2 with `--final`, once per seed, on a frozen config.
2. **No data leakage:** record-wise splits, extra databases are **training data only**.
3. **Physical millivolts only,** no per-window normalisation. Input scaling uses fixed power-of-two
   divisors so it maps to shifts in hardware.
4. **Strictly causal features** (pre-RR only). A one-beat delay was tested and did not help.
5. **Report mean ± std over seeds,** and keep negative results.

Disclosure: an early sweep looked at DS2 when choosing dV = 0.1 mV. It was re-derived on DS1 only and
agrees; the old files are in `results/archive_first_run/` and must not be cited.

## Next steps (suggested order)

1. **Run on the GPU laptop** (RTX 5060 → needs a CUDA 12.8+ PyTorch build). Dev run with SVDB as
   extra training data: `python scripts/train_student.py --delta 0.1 --svdb-dir data/svdb --seeds 0`
   (DS1 validation only). Expect a big speed-up over the 2-core cloud box.
2. **Fix the teacher.** It scores *below* the student (87.9 % vs 93.7 %) and overfits (train loss
   ≈ 0.001), so distillation is not helping. Try stronger regularisation / augmentation / richer inputs.
3. **Ablation table** for the paper: template on/off, gain-normalisation on/off, dV, SVDB on/off,
   ideally with the deep student (not just gradient boosting).
4. **Width / size sweep** (6, 9, 12, 24, 48 KB): is capacity a bottleneck? (Hypothesis: no.)
5. **Final multi-seed DS2 run** (`--final`) once the configuration is frozen.
6. **Size-capped tree-ensemble baseline** (XGBoost/LightGBM ≤ 12 KB) as a second model family.
7. **Literature search + re-run published baselines under the same split** before any SOTA claim.
8. **Cross-dataset test** (train MIT-BIH, test e.g. INCART) — reviewers will ask.
9. **Energy-per-beat comparison vs a duty-cycled MCU** — the original "clock-gating saves power"
   thesis is unproven because static power dominates (≈ 76 µJ idle vs ≈ 0.24 µJ compute per beat on
   the iCE40 plan). Decide the compute platform only after this.

## Gotchas we hit

* **Stale bytecode:** if you edit a module without changing its file size, a stale `.pyc` can be
  reused on some filesystems (we got wrong feature numbers once). Use `PYTHONDONTWRITEBYTECODE=1`.
* **Network allowlist:** PhysioNet files redirect to `physionet-open.s3.amazonaws.com`; that host had
  to be allowed in the sandbox.
* **Windows/Anaconda:** install torch with `--index-url https://download.pytorch.org/whl/cu128`.
* The analytic probes use scikit-learn gradient boosting; the deep model needs torch.

## Open design questions

* Compute platform: low-power M4-class MCU vs the iCE40 FPGA (static ≈ 75 µA core). Back-of-envelope
  says M4 + int8 SIMD kernels ≈ 10–20 µJ/beat; verify with datasheets.
* The `INA333` (≈ 90 µW) dominates total power; a lower-power in-amp / ECG AFE is the biggest lever.
* Spike handling must not wake the CPU per spike (~55 spikes/s): use a counting/timestamp peripheral.
* S and F are weak for a *causal* detector; more training patients is the most promising lever
  (SVDB already shows S F1 0.257 → 0.323 in DS1 CV).
