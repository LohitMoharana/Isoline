# Project Handoff: Sub-Milliwatt Event-Driven Hardware ECG Inference System

## 1. Objective & Research Framing
We are designing and building an end-to-end, sub-milliwatt (< 1 mW total board power) hardware-software co-designed ECG arrhythmia inference system on a custom 4-layer PCB, targeted as a systems-and-circuits research project (e.g., IEEE TBioCAS, TCAS-II, DATE, or MLSys).

**Core Architectural Thesis:** Standard edge runtimes (TFLite Micro, CMSIS-NN) waste power by running dense, frame-based inference continuously across the >80% of the cardiac cycle that is flat isoelectric baseline. Instead, we pair a nanopower analog front-end (AFE) and dual level-crossing comparators with a clock-gated FPGA inference engine so digital compute stays completely unclocked until a heartbeat crosses voltage thresholds.

---

## 2. Target Hardware & Silicon Stack (Hand-Solderable, Zero BGA)
* **Power Tree (~0.3 uW quiescent):**
  * Two TI `TPS62840` ultra-low-Iq (60 nA) step-down buck converters generating `+1.8V_ANA`, `+1.8V_IO`, and `+1.2V_CORE` from a LiPo or CR2032 cell.
  * TI `TPS22916` (10 nA leakage) load switches to power-gate external SPI configuration flash after FPGA boot and isolate optional telemetry radios.
* **Nanopower Analog Front-End (~92 uW continuous at 1.8V):**
  * **Instrumentation Amplifier:** TI `INA333` (VSSOP-8, 50 uA, 100 dB CMRR), fixed gain $G = 100$, biased around a $0.9\text{ V}$ (`Vdd/2`) virtual ground.
  * **Conditioning & Right-Leg Drive (RLD):** TI `TLV8802` dual nanopower op-amp (320 nA/ch) implementing a 1st-order $0.5\text{ Hz}$ DC-servo high-pass filter (or $0.25\text{ Hz}$ if needed for ST-segment integrity), a 2nd-order $40\text{ Hz}$ Butterworth low-pass filter, and an active RLD loop.
  * **Asynchronous Event Digitizer:** Dual TI `TLV3691` nanopower comparators (75 nA each) configured as a level-crossing delta modulator emitting `+1` (up-crossing) and `-1` (down-crossing) spikes whenever the conditioned ECG moves by $\pm\Delta V$, waking the digital clock tree on QRS onset.
* **Custom Acceleration Engine (~95 uW idle / ~480 uW active burst):**
  * **Target Silicon:** Lattice `iCE40UP5K-SG48I` (48-pin QFN, 5,280 LUTs, 120 kbit Single-Port SRAM, 8 hardware 16x16 MACs, 75 uA static core current at 1.2V).
  * **Execution Model:** High-frequency oscillator (`SB_HFOSC`) stays gated off during idle intervals; `TLV3691` comparator interrupts wake the engine to burst-compute classification at 6–12 MHz in < 500 us before gating the clock back off.
* **Optional Supervisor / Radio:** `nRF52840` or `STM32U0` kept in deep sleep (< 2.5 uW) and woken via GPIO only when an arrhythmia is detected.

---

## 3. Datasets & Non-Negotiable Ingestion Rules
1. **Datasets:**
   * **MIT-BIH Arrhythmia (`mitdb`, 360 Hz):** Primary beat-level classification benchmark (AAMI classes `N`, `S`, `V`, `F`; 4 paced records excluded).
   * **European ST-T (`edb`, 250 Hz):** Phase-distortion stress test verifying the analog high-pass filter does not warp ischemic ST-segment shifts.
   * **PTB Diagnostic (`ptbdb`, 1000 Hz):** Single-lead (`Lead II`) myocardial infarction vs. healthy morphological benchmark.
   * **PhysioNet/CinC 2017 (`challenge2017`, 300 Hz):** Rhythm classification (`Normal`, `AFib`, `Other`) and real-world artifact rejection (`Noisy` class).
2. **Strict Physical Millivolts:** Convert all records into true physical millivolts using native `wfdb` `adc_gain` and `baseline` metadata. **Never** apply per-window z-score normalization, because physical comparators operate on fixed voltage thresholds.
3. **Unified Timebase & Splits:** Resample all records to $360\text{ Hz}$ using zero-phase polyphase filtering (`scipy.signal.resample_poly`). Enforce strict subject-disjoint splits (**De Chazal DS1 train / DS2 test** for MIT-BIH; subject-ID splits for STT and PTB) to prevent patient anatomy leakage.

---

## 4. Algorithm & RTL-Exact Quantization Spec
* **Hardware-Exact Features (Zero Floating-Point):**
  * **Morphology Channels (3 x 32 bins):** Temporal histograms of `+1` up-spikes and `-1` down-spikes across a 256-sample beat window (~711 ms), plus a 3rd channel from a single 8-bit RTL leaky integrator (`acc[n] = acc[n-1] - (acc[n-1] >> 4) + spike[n]`) to preserve slow P/T/ST wave envelopes.
  * **Causal Gain Adaptation:** Exponentially weighted moving average of spike energy per beat using bit-shifts (`ewma = ewma - (ewma >> 3) + (beat_spikes >> 3)`).
  * **Causal RR-Interval Timing:** 16-bit timer ticks scaled to `Q3.5` fixed-point (`1 sec = 32 counts`) tracking `pre_RR`, `rr_ewma`, and instantaneous delta (`pre_RR - prev_pre_RR`) to separate premature Supraventricular (`S`) beats and `AFib` without waiting for future beats.
* **Model & Silicon Budget:**
  * Compact Dilated Depthwise-Separable 1D-CNN (`StudentSiliconTCN`) trained via Teacher–Student knowledge distillation (from a full-precision `SE-ResNet1D` teacher) + Class-Balanced Focal Loss + random comparator trigger jitter augmentation ($\pm 5$ samples).
  * **Hard Caps:** Total model footprint **<= 12 KB** (`INT8` weights, `INT32` accumulators, power-of-two bit-shift rescaling only, BatchNorm folded into weights prior to export).
* **PoC Exit Gates:**
  * Overall inter-patient accuracy **>= 95.5%** on MIT-BIH DS2 (`V` F1 >= 0.91), idle spike sparsity **> 80%**, and **< 2% F1 degradation** on European ST-T ischemic episodes post-AFE.

---

## 5. Seven-Phase Execution Roadmap
1. **Phase 1 — Unified Data & AFE Emulation (Python):** Load all 4 datasets in physical mV, enforce inter-patient splits, emulate the 1.8V AFE + `TLV3691` level crossings, and sweep $\Delta V$.
2. **Phase 2 — SOTA Distillation & QAT Verification (Python):** Train and quantize the <= 12 KB `INT8` model, clear the >= 95.5% accuracy and > 80% sparsity gates, and export quantized weights (`.hex`) plus pathological ECG `.pwl` files.
3. **Phase 3 — Analog Circuit Verification (`ngspice` / Multisim):** Simulate the `INA333` + `TLV8802` + `TLV3691` + `TPS62840` schematic driven by Python `.pwl` waveforms through asymmetrical $10\text{ k}\Omega$ / $47\text{ k}\Omega$ electrode resistors and $50\text{ Hz}$ common-mode hum.
4. **Phase 4 — RTL Design & Cosimulation (Verilog / Verilator / Yosys):** Build the clock-gated inference core for `iCE40UP5K`, verify bit-exact parity against Python `INT8` predictions, and confirm < 4,200 LUTs / < 100 kbit SPRAM at 12 MHz.
5. **Phase 5 — Isolated Hardware-in-the-Loop (HIL) Bench:** Build an MCU + 12-bit SPI DAC streamer locked to a $360\text{ Hz}$ hardware timer, a $159\text{ Hz}$ RC smoothing filter, and a $1000:1$ attenuator ($100\text{ k}\Omega$ / $100\text{ }\Omega$), powered via USB isolator or battery to replay $1\text{ mV}$ clinical records without ground loops.
6. **Phase 6 — Rev 1 Custom PCB (AFE + Power-Tree Carrier):** Spin a 4-layer KiCad board with driven guard rings and per-rail current-sense jumpers mated to an `iCE40UP5K` breakout board to validate real analog signal integrity and microwatt rail currents.
7. **Phase 7 — Rev 2 Unified Monolithic PCB:** Merge the `iCE40UP5K` onto a single compact 4-layer board and log empirical latency (us), energy per beat (uJ), and post-AFE clinical F1.

---

## 6. Current Status & Where to Start
We have finalized the system architecture, silicon BOM, dataset rules, and 7-phase roadmap. We are currently at **Phase 1 & Phase 2 (Python PoC)**—before writing Verilog or drawing schematics, we need to build and verify the complete, runnable Python pipeline that loads the datasets in physical millivolts, emulates the AFE + leaky integrator + causal RR features, trains the QAT student model, and confirms it clears the >= 95.5% inter-patient accuracy and > 80% sparsity gates.