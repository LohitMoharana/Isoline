# Power budget (estimate, 3.0 V, ~1.2 beats/s, ~55 comparator events/s)

Datasheet figures are typical values from the vendors' product pages (Nov 2026). Compute energy is an
**estimate** from cycle counts, not yet measured on silicon.

## Architecture A: discrete event-driven front end (the Isoline concept)

| Block | Part | Current | Power @ 3 V |
|---|---|---|---|
| In-amp | INA333 (50 µA typ, 1.8–5.5 V) | 50 µA | 150 µW |
| Servo, gain and LPF op amps | 3 × TLV8541 (0.5 µA each) | 1.5 µA | 4.5 µW |
| Comparators | 2 × TLV3691 (75 nA each) | 0.15 µA | 0.45 µW |
| MCU sleep | STM32L4 Stop 2 (≈1 µA) | 1 µA | 3 µW |
| Event handling | ~55 wake-ups/s × ~50 nJ (est.) | — | ≈3 µW |
| Classification | ~25 k cycles/beat × 252 pJ/cycle (84 µA/MHz) ≈ 6 µJ/beat | — | ≈8 µW |
| **Total** | | **≈56 µA** | **≈170 µW** |

Running the in-amp at 1.8 V instead of 3 V roughly halves the dominant term (≈90 µW), so the total is about 110 µW.

## Architecture B: integrated ECG AFE + MCU (conventional)

| Block | Part | Power |
|---|---|---|
| ECG AFE with ADC + R-R detection | MAX30003 (85 µW at 1.1 V; 32-word FIFO, wake every 256 ms) | 85 µW |
| MCU (sleep + FIFO reads + software features + classification) | STM32L4 | ≈15–25 µW (est.) |
| **Total** | | **≈100–110 µW** |

## What this means (honest)

* **The analog front end dominates** in both architectures (≥ 80 % of the power).
* The event-driven idea saves **digital** energy (no ADC, compute asleep about 90 % of the time), but the digital side is
  only a few µW. **With off-the-shelf parts, Architecture A does not use less power than a modern integrated ECG AFE.**
  It is about equal at best, at 1.8 V.
* Where the concept pays off: (1) the tiny classifier and event features work with either front end; (2) a **custom
  analog front end or ASIC** with a nanowatt in-amp would let the comparator approach win clearly. That is future work.
* Battery life on 2 × AAA (≈1000 mAh typical alkaline, to verify) at ≈56 µA is ≈ 2 years (ideal; self-discharge
  and derating will reduce it). Both architectures meet the original 6-month target comfortably.
