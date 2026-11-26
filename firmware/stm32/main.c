/* Isoline application for the NUCLEO-L432KC on the Isoline AFE carrier board (hardware/kicad).
 * Register-level (CMSIS only, no HAL). Pin map (hardware/kicad/gen/make_netlist.py):
 *   PA4  DAC1_OUT1 -> TH_UP  (upper comparator threshold)      PA12 <- EV_UP  (TLV3691 #1 output, EXTI12)
 *   PA5  DAC1_OUT2 -> TH_DN  (lower comparator threshold)      PB0  <- EV_DN  (TLV3691 #2 output, EXTI0)
 *   PB1  -> FAST_RESTORE (servo fast-restore switch)            PA2  -> USART2_TX (ST-LINK virtual COM port)
 * Timebase: LPTIM1 clocked from LSE 32.768 kHz, period 91 -> 360.09 Hz slot tick (= the 360 Hz model timebase).
 *
 * Level crossing (matches isoline.afe): the DAC holds ref +/- step. A rising edge on EV_UP means the AFE output
 * crossed ref + step: count one up event and move both thresholds up by one step. Steep QRS edges can cross several
 * steps inside one slot; the comparator output then stays high, so each slot tick also re-checks the outputs and keeps
 * stepping (with a settle delay for the TLV3691's 24 us response) while they are still asserted.
 *
 * TODO (documented in firmware/README.md): Stop 2 instead of Sleep between slots (DAC sample-and-hold), VDDA
 * tracking of the DAC step via VREFINT, event-timestamp mode without the periodic tick.
 */
#include <stdint.h>
#include "stm32l432xx.h"
#include "isoline_stream.h"
#include "isoline_model.h"

#define DAC_STEP 41u            /* 30 mV at VDDA = 3.0 V: 0.1 mV input x gain 300 (hardware/design.py) */
#define DAC_MID 2048u           /* VREF = VDDA / 2 */
#define SETTLE_LOOPS 40         /* ~30 us at 4 MHz: TLV3691 response time is 24 us */
#define MAX_STEPS_PER_SLOT 16

static volatile uint16_t ev_up, ev_dn;          /* events in the current slot (ISR -> tick) */
static volatile int32_t ref_code = DAC_MID;
static volatile uint8_t slot_ready;
static volatile uint16_t slot_up, slot_dn;
static iso_stream S;

static void dac_write(void) {
    int32_t up = ref_code + DAC_STEP, dn = ref_code - DAC_STEP;
    if (up > 4095) up = 4095;
    if (dn < 0) dn = 0;
    DAC1->DHR12R1 = (uint32_t)up;
    DAC1->DHR12R2 = (uint32_t)dn;
}

static inline int ev_up_level(void) { return (GPIOA->IDR >> 12) & 1u; }
static inline int ev_dn_level(void) { return (GPIOB->IDR >> 0) & 1u; }
#ifdef ISO_HOST_SIM   /* virtual-HIL build (sim/hil_sim.c): the simulator stands in for analog time and the UART */
static void settle(void) { sim_settle(); }
#else
static void settle(void) { for (volatile int i = 0; i < SETTLE_LOOPS; i++) { } }
#endif

void EXTI15_10_IRQHandler(void) {                /* EV_UP rising edge */
    if (EXTI->PR1 & EXTI_PR1_PIF12) { EXTI->PR1 = EXTI_PR1_PIF12; ev_up++; ref_code += DAC_STEP; dac_write(); }
}

void EXTI0_IRQHandler(void) {                    /* EV_DN rising edge */
    if (EXTI->PR1 & EXTI_PR1_PIF0) { EXTI->PR1 = EXTI_PR1_PIF0; ev_dn++; ref_code -= DAC_STEP; dac_write(); }
}

void LPTIM1_IRQHandler(void) {                   /* 360 Hz slot boundary */
    if (LPTIM1->ISR & LPTIM_ISR_ARRM) {
        LPTIM1->ICR = LPTIM_ICR_ARRMCF;
        for (int k = 0; k < MAX_STEPS_PER_SLOT; k++) {   /* steep edges: comparators still asserted -> more steps */
            settle();
            if (ev_up_level()) { ev_up++; ref_code += DAC_STEP; dac_write(); }
            else if (ev_dn_level()) { ev_dn++; ref_code -= DAC_STEP; dac_write(); }
            else break;
        }
        slot_up = ev_up; slot_dn = ev_dn; ev_up = 0; ev_dn = 0; slot_ready = 1;
    }
}

static void uart_init(void) {                    /* USART2 on PA2 (AF7), 115200 baud from MSI 4 MHz */
    RCC->APB1ENR1 |= RCC_APB1ENR1_USART2EN;
    GPIOA->MODER = (GPIOA->MODER & ~(3u << 4)) | (2u << 4);
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~(0xFu << 8)) | (7u << 8);
    USART2->BRR = 4000000u / 115200u;
    USART2->CR1 = USART_CR1_TE | USART_CR1_UE;
}

static void uart_puts(const char *s) {
#ifdef ISO_HOST_SIM
    while (*s) sim_uart_putc(*s++);
#else
    while (*s) { while (!(USART2->ISR & USART_ISR_TXE)) { } USART2->TDR = (uint8_t)*s++; }
#endif
}

static void uart_putu(uint32_t v) {
    char b[11]; int i = 10; b[i] = 0;
    do { b[--i] = (char)('0' + v % 10u); v /= 10u; } while (v);
    uart_puts(&b[i]);
}

static void hw_init(void) {
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN | RCC_AHB2ENR_GPIOBEN;
    RCC->APB1ENR1 |= RCC_APB1ENR1_DAC1EN | RCC_APB1ENR1_LPTIM1EN | RCC_APB1ENR1_PWREN;
    RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;
    /* analog mode on PA4/PA5 (DAC outputs) - reset default is analog on L4; make it explicit */
    GPIOA->MODER |= (3u << 8) | (3u << 10);
    /* PA12, PB0 inputs (comparator outputs are push-pull, no pull needed) */
    GPIOA->MODER &= ~(3u << 24);
    GPIOB->MODER &= ~(3u << 0);
    /* PB1 output: fast-restore switch */
    GPIOB->MODER = (GPIOB->MODER & ~(3u << 2)) | (1u << 2);
    /* DAC: both channels, buffer on */
    DAC1->CR = DAC_CR_EN1 | DAC_CR_EN2;
    dac_write();
    /* EXTI12 <- PA12, EXTI0 <- PB0, rising edges */
    SYSCFG->EXTICR[3] = (SYSCFG->EXTICR[3] & ~SYSCFG_EXTICR4_EXTI12) | SYSCFG_EXTICR4_EXTI12_PA;
    SYSCFG->EXTICR[0] = (SYSCFG->EXTICR[0] & ~SYSCFG_EXTICR1_EXTI0) | SYSCFG_EXTICR1_EXTI0_PB;
    EXTI->RTSR1 |= EXTI_RTSR1_RT12 | EXTI_RTSR1_RT0;
    EXTI->IMR1 |= EXTI_IMR1_IM12 | EXTI_IMR1_IM0;
    NVIC_EnableIRQ(EXTI15_10_IRQn); NVIC_EnableIRQ(EXTI0_IRQn);
    /* LSE on (Nucleo-L432KC fits a 32.768 kHz crystal; verify for your board revision) */
    PWR->CR1 |= PWR_CR1_DBP;
    RCC->BDCR |= RCC_BDCR_LSEON;
    while (!(RCC->BDCR & RCC_BDCR_LSERDY)) { }
    RCC->CCIPR = (RCC->CCIPR & ~RCC_CCIPR_LPTIM1SEL) | (3u << RCC_CCIPR_LPTIM1SEL_Pos);   /* LPTIM1 clock = LSE */
    LPTIM1->IER = LPTIM_IER_ARRMIE;
    LPTIM1->CR = LPTIM_CR_ENABLE;
    LPTIM1->ARR = 90;                                   /* 91 LSE ticks = 2.777 ms = 360.09 Hz */
    LPTIM1->CR |= LPTIM_CR_CNTSTRT;
    NVIC_EnableIRQ(LPTIM1_IRQn);
}

#ifdef ISO_HOST_SIM
static void delay_ms(uint32_t ms) { (void)ms; }     /* fast restore is not part of the steady-state AFE model */
#else
static void delay_ms(uint32_t ms) { for (volatile uint32_t i = 0; i < ms * 500u; i++) { } }
#endif

int main(void) {
    hw_init();
    uart_init();
    GPIOB->BSRR = 1u << 1; delay_ms(1000); GPIOB->BRR = 1u << 1;    /* fast-restore the 0.05 Hz servo for 1 s */
    iso_stream_init(&S);
    uart_puts("isoline ready\r\n");
    static const char cls[4] = {'N', 'S', 'V', 'F'};
    iso_beat out[2];
    for (;;) {
        __WFI();                                         /* Sleep until the next interrupt */
        if (!slot_ready) continue;
        slot_ready = 0;
        int n = iso_stream_push(&S, slot_up, slot_dn, out, 2);
        for (int i = 0; i < n; i++) {
            if (out[i].have_feat != 1) continue;
            int c = isoline_classify(out[i].feat, out[i].af_valid ? out[i].af : 0, 1);
            char tail[4] = {' ', cls[c], '\r', 0};
            uart_putu((uint32_t)out[i].r); uart_puts(tail); uart_puts("\n");
        }
    }
}
