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

## Update 2026-11-25

* Round 2 (see README): INCART extra data, student without teacher, selection by macro-F1 -> DS2 accuracy 92.3 ± 1.3 %,
  macro-F1 0.530, S F1 0.270, V F1 0.868. S improves, accuracy and V drop slightly. DS2 has now been used for two
  frozen final runs; any further DS2 number must be reported as a new, additional look.
* Negative DS1 results are in `results/exp_ideas_log.txt` (second lead, class scaling, dense waveform, multi-scale dV).
* Runs happen on a 2-core cloud CPU; the RTX laptop is not reachable from the cloud session, so GPU runs must be started by hand
  on Windows (`pip install torch --index-url https://download.pytorch.org/whl/cu128`).
* Next: tune the INCART mix (N subsampling, weight), try the SVDB+INCART combination on the student, stronger
  regularisation (loss is ~0.002), and decide how to pick checkpoints on a larger validation set than 4 records.

### Round 3 (same day)
DS1-CV study of extra-data mixes, model size and finer bins: all give 94.6–94.9 %; see README. **No new DS2 look was
taken.** The 96 % target was *not* reached. Suggested next diagnostic: a dense-waveform upper bound with the same
template/gain features (does the encoder, or the features, cap accuracy?), then a front-end change that preserves
P-wave/morphology detail, evaluated first on DS1 CV.

### Round 4 (2026-11-25 night)
Frozen GBT candidate (long RR + online clusters + INCART) -> **DS2 94.76 ± 0.08 %**, macro-F1 0.605, S F1 0.33,
V F1 0.956. Third DS2 look; nothing tuned on it. Next: S detection (DS1 CV only), and porting the new features to the
int8 student / a size-capped tree model.

### Round 5 (2026-11-26, ~00:30)
Two-stage prediction memory + S scale 6, frozen on DS1 CV + external SVDB -> **DS2 93.00 ± 0.34 %** (worse). Fourth DS2
look. **Best honest result remains round 4: 94.76 ± 0.08 %.** Do not take another DS2 look without a genuinely new
idea and a dev set shown to track DS2; consider reporting DS2 with a correction for the number of looks.

### Round 6 (2026-11-26 morning)
Clean-protocol replica of the 96.14 % RF paper scores 95.39 % on DS2 (dense ECG, not event-driven). The event port
matches it on DS1 after two fixes: a mean-baseline for widths and an HPF of 0.05 Hz instead of 0.5 Hz. The pre-registered DS1-LOPO
rule picked round 4 (already evaluated, 94.76 %), so no new DS2 look. Open question for later: the 0.05 Hz event
port was never evaluated on DS2. Any future look must be justified on DS1 first and disclosed.

### Rounds 7-8 (2026-11-26)
Round 7: 5.2 KB tiny GBT (round-4 features + INCART) -> DS2 93.27 %, V F1 0.947, S F1 0.31. Round 8: MIT-only + port
features + HPF 0.05 + next-RR, chosen on DS1 LOPO + SVDB -> DS2 92.65 % (worse). DS2 has now been used 9 times.
**Stop iterating against DS2.** Next real step: a held-out test set nobody has looked at (e.g. a different database, or
recordings from the PCB), and report every DS2 number with the number of evaluations behind it.

### Reliability study (2026-11-26)
Noise stress and cross-database results for the frozen 5.2 KB model are in the README. Main weakness: false alarms
(V precision) under electrode-motion noise and on other databases. Next ideas: alarm smoothing / burst logic, AF flag
(MIT-BIH AFDB), bit-exact C export + cycle count, LTspice AFE power.

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
