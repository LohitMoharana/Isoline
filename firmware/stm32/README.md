# STM32L432KC application (Nucleo-L432KC on the Isoline AFE carrier)

Register-level CMSIS firmware: no HAL, no RTOS.

| Pin | Function |
|---|---|
| PA4 / PA5 | DAC1 ch1/ch2: upper/lower comparator thresholds (ref ± 30 mV = 41 LSB at 3.0 V) |
| PA12 / PB0 | EV_UP / EV_DN comparator outputs, EXTI rising edge |
| PB1 | Fast-restore switch, held high for 1 s at power-up |
| PA2 | USART2 TX → ST-LINK virtual COM, 115200 8N1; prints `<sample index> <N/S/V/F>` per beat |

The LPTIM1 slot tick runs from the LSE: 91 ticks give 360.09 Hz. On each tick the comparators are re-checked, so steep QRS edges that cross several steps in one slot are counted. Each slot's counts then go to `iso_stream_push` (bit-exact with the Python pipeline) and each emitted beat to `isoline_classify` (int8 leaves).

## Build
```
git clone https://github.com/STMicroelectronics/cmsis_device_l4
git clone https://github.com/STMicroelectronics/cmsis_core
make CMSIS_DEVICE=path/to/cmsis_device_l4 CMSIS_CORE=path/to/cmsis_core/CMSIS/Core/Include
```
arm-none-eabi-gcc 13.2, -O2: **66 KB flash, 25 KB RAM** of the L432KC's 256 KB / 64 KB. The flash includes reference float32 leaves and a non-compacted tree layout. The compacted tiny model is 5.2 KB.
The `_close/_lseek/_read/_write not implemented` link warnings come from newlib exit handling and are harmless, because `main` never returns.

## Not done yet (honest list)
- Not run on real hardware. Timing, DAC settling and EXTI latency are unverified.
- Sleep mode only. Stop 2 with DAC sample-and-hold is the low-power target.
- DAC step assumes VDDA = 3.0 V. It should track VREFINT as the batteries drain.
- The core's clock comes from the default MSI 4 MHz. The cycle budget per beat has not been measured on target.

## Virtual hardware-in-the-loop (sim/)
`sim/hil_sim.c` compiles the **unmodified `main.c`** (EXTI and LPTIM handlers, DAC stepping, re-check loop, streaming features, int8 classifier) against a mock register file. The analog side it simulates:
- **AFE output:** the ngspice-derived circuit response with rail clipping (`hardware/sim/spice_frontend.py`), interpolated at 64 sub-steps per slot.
- **DAC:** 12-bit, `code × VDDA / 4095`, so the real step is 30.04 mV, not the ideal 30.00 mV.
- **Comparators:** two TLV3691s with 17 mV hysteresis and optional input offsets.
- **Timer:** the real LSE slot of 91/32768 s (360.09 Hz). Its 0.025% drift is undone only when scoring.

The only `#ifdef ISO_HOST_SIM` lines in `main.c` replace the UART register write, the settle spin-loop and the 1 s fast-restore delay. `sim/hil_run.py` scores the UART beat list against the MIT-BIH annotations: EC57 ±150 ms matching, AAMI classes, and missed beats counted as errors for end-to-end accuracy.

DS1 runs were used for development (detection: Se 99.12%, +P 98.20%, against the Python detector's 99.09% / 97.65%). DS2 ran once in four conditions fixed beforehand, with nothing changed afterwards (DS2 look #13).

| DS2 (held-out patients), firmware in the loop | Matched acc. | End-to-end | Det. Se | Det. +P | V Se / +P | S Se / +P |
|---|---|---|---|---|---|---|
| Nominal (VDDA 3.0 V, no offsets) | **93.78%** | **93.49%** | 99.69% | 99.08% | 96.4 / 84.1 | 23.6 / 52.1 |
| Comparator offsets +5 / −5 mV | 94.05% | 93.72% | 99.64% | 99.20% | 96.0 / 86.2 | 23.3 / 54.2 |
| Battery sag, VDDA 2.6 V (step 26 mV) | 93.74% | 93.44% | 99.68% | 98.89% | 95.9 / 78.2 | 23.5 / 52.9 |
| Fresh cells, VDDA 3.2 V (step 32 mV) | 94.13% | 93.86% | 99.71% | 99.11% | 96.1 / 86.5 | 22.3 / 54.0 |
| *Python pipeline, tiny + AF, detected beats (5-seed mean)* | *93.67%* | *93.47%* | *99.78%* | *98.67%* | | |

So the firmware reproduces the Python system (one seed against the five-seed mean), and DAC/battery/offset errors move accuracy by less than 0.4 points. S-beat sensitivity is the weak spot, as everywhere in this project; it is a model limit, not a firmware one.
**Not modelled:** electrode noise and motion, the fast-restore transient, DAC settling glitches, interrupt latency, and real silicon.
