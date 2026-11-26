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
