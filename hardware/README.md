# Isoline hardware

Status: **design + simulation**. Nothing has been built yet.

## Analog front end (proposal) and its ngspice check

```
electrodes ─► INA333 (G=10, 50 µA) ─► ×30 gain (TLV8541) ─► 40 Hz Sallen-Key LPF (TLV8541) ─► 2 comparators ─► MCU
                 ▲ REF                                                                       (±30 mV steps)
                 └── DC servo integrator (TLV8541, 10 MΩ / 330 nF → 0.05 Hz high-pass)
```

* Total gain 300, so a 0.1 mV input step is **30 mV at the comparators**, which clears the TLV3691's ±17 mV hysteresis.
  At gain 100, a 10 mV step would be swallowed by the hysteresis. This is a design constraint the simulation-only model did not show.
* High-pass at **0.05 Hz**: the algorithm study showed that a 0.5 Hz corner distorts QRS shape and costs accuracy.
* `sim/make_afe_sim.py` writes `sim/afe.cir`, runs ngspice on a real MIT-BIH segment (record 101, with a 100 mV electrode
  offset) and compares the circuit with the Python AFE model used for every accuracy number:

| check | result |
|---|---|
| waveform correlation (input-referred, after 2 s) | **0.993** |
| RMS difference | **25 µV** (a quarter of one 0.1 mV comparator step) |
| level-crossing events, circuit vs model | **433 vs 435 (−0.5 %)** |
| output swing | 0.30 – 1.99 V on a 0–3 V supply (headroom OK) |

* **Known issue:** at power-up the 0.05 Hz servo takes several seconds to cancel electrode offset, and the gain stage
  saturates meanwhile. The simulation starts the servo pre-charged, and the board needs a fast-restore switch (the AD8233
  integrates one). That is a design to-do.
* Op amps are behavioural models (gain, GBW and rail limits from datasheets). Vendor SPICE models should replace them before tape-out.

See [POWER_BUDGET.md](POWER_BUDGET.md) for the power analysis and the architecture decision it raises.
