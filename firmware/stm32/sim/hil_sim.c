/* Virtual hardware-in-the-loop: runs the unmodified firmware (main.c: ISRs, DAC stepping, re-check loop, streaming
 * features, classifier) against a simulated analog front end.
 *
 *   input : float32 file, AFE output voltage (V, the comparator input node) at 360 Hz, from the ngspice-derived
 *           circuit response (hardware/sim/spice_frontend.py); linearly interpolated between samples (40 Hz band limit)
 *   DAC   : 12-bit, Vout = code * VDDA / 4095 (so the 41-LSB step is 30.04 mV, not the ideal 30.00 mV)
 *   comparators (TLV3691): output toggles with hysteresis H around the threshold, plus a per-comparator input offset
 *   time  : LPTIM slot = 91 / 32768 s (360.09 Hz, the real LSE divider), SUB sub-steps per slot; EXTI fires on a
 *           0->1 transition of a comparator output seen at a sub-step; the slot tick runs the firmware's re-check loop
 *
 * usage : hil_sim wave.f32 [H_mV=17] [off_up_mV=0] [off_dn_mV=0] [VDDA=3.0] [SUB=64]   -> firmware UART on stdout
 */
#include <stdio.h>
#include <stdlib.h>
#include "stm32l432xx.h"
#define main fw_main
#include "../main.c"
#undef main

GPIO_TypeDef sim_gpioa, sim_gpiob; RCC_TypeDef sim_rcc; DAC_TypeDef sim_dac; EXTI_TypeDef sim_exti;
SYSCFG_TypeDef sim_syscfg; PWR_TypeDef sim_pwr; LPTIM_TypeDef sim_lptim; USART_TypeDef sim_usart;

static float *wave; static long nwave;
static double H = 0.017, OFF_UP = 0, OFF_DN = 0, VDDA = 3.0; static int SUB = 64;
static const double TSLOT = 91.0 / 32768.0;
static long slot_k; static int sub_i; static int out_up, out_dn;

static double v_at(double t) {                       /* AFE output at time t (s) */
    double x = t * 360.0; long i = (long)x;
    if (i >= nwave - 1) return wave[nwave - 1];
    double f = x - (double)i;
    return wave[i] * (1 - f) + wave[i + 1] * f;
}
static double now(void) { return ((double)slot_k + (double)sub_i / SUB) * TSLOT; }

static void comparators(double v) {                   /* update both outputs with hysteresis; returns via globals */
    double th_up = sim_dac.DHR12R1 * VDDA / 4095.0 + OFF_UP, th_dn = sim_dac.DHR12R2 * VDDA / 4095.0 + OFF_DN;
    if (v > th_up + H / 2) out_up = 1; else if (v < th_up - H / 2) out_up = 0;
    if (v < th_dn - H / 2) out_dn = 1; else if (v > th_dn + H / 2) out_dn = 0;
    sim_gpioa.IDR = (sim_gpioa.IDR & ~(1u << 12)) | ((uint32_t)out_up << 12);
    sim_gpiob.IDR = (sim_gpiob.IDR & ~1u) | (uint32_t)out_dn;
}

void sim_settle(void) { comparators(v_at(now())); }  /* comparator has responded after the firmware's settle delay */
void sim_uart_putc(char c) { putchar(c); }

void sim_wfi(void) {                                  /* advance analog time to the next slot tick */
    for (; sub_i < SUB; sub_i++) {
        int pu = out_up, pd = out_dn;
        comparators(v_at(now()));
        if (!pu && out_up && (sim_exti.IMR1 & EXTI_IMR1_IM12)) { sim_exti.PR1 |= EXTI_PR1_PIF12; EXTI15_10_IRQHandler(); sim_exti.PR1 = 0; comparators(v_at(now())); }
        if (!pd && out_dn && (sim_exti.IMR1 & EXTI_IMR1_IM0)) { sim_exti.PR1 |= EXTI_PR1_PIF0; EXTI0_IRQHandler(); sim_exti.PR1 = 0; comparators(v_at(now())); }
    }
    sub_i = 0; slot_k++;
    if (now() * 360.0 >= nwave - 1) { fflush(stdout); exit(0); }
    sim_lptim.ISR |= LPTIM_ISR_ARRM; LPTIM1_IRQHandler(); sim_lptim.ISR = 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: hil_sim wave.f32 [H_mV] [off_up_mV] [off_dn_mV] [VDDA] [SUB]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); nwave = ftell(f) / 4; fseek(f, 0, SEEK_SET);
    wave = malloc(sizeof(float) * nwave);
    if (fread(wave, 4, nwave, f) != (size_t)nwave) return 1;
    fclose(f);
    if (argc > 2) H = atof(argv[2]) * 1e-3;
    if (argc > 3) OFF_UP = atof(argv[3]) * 1e-3;
    if (argc > 4) OFF_DN = atof(argv[4]) * 1e-3;
    if (argc > 5) VDDA = atof(argv[5]);
    if (argc > 6) SUB = atoi(argv[6]);
    sim_rcc.BDCR = RCC_BDCR_LSERDY; sim_usart.ISR = USART_ISR_TXE;
    return fw_main();
}
