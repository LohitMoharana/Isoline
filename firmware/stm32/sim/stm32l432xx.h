/* Host-simulation stand-in for the CMSIS device header: same register names, plain structs in RAM.
 * Only what main.c touches. Bit values follow RM0394 where it matters to the simulator (pin bits); the rest only
 * need to be distinct. Used by hil_sim.c; never by the target build. */
#ifndef SIM_STM32L432XX_H
#define SIM_STM32L432XX_H
#include <stdint.h>
typedef struct { volatile uint32_t MODER, IDR, BSRR, BRR, AFR[2]; } GPIO_TypeDef;
typedef struct { volatile uint32_t AHB2ENR, APB1ENR1, APB2ENR, BDCR, CCIPR; } RCC_TypeDef;
typedef struct { volatile uint32_t CR, DHR12R1, DHR12R2; } DAC_TypeDef;
typedef struct { volatile uint32_t PR1, RTSR1, IMR1; } EXTI_TypeDef;
typedef struct { volatile uint32_t EXTICR[4]; } SYSCFG_TypeDef;
typedef struct { volatile uint32_t CR1; } PWR_TypeDef;
typedef struct { volatile uint32_t ISR, ICR, IER, CR, ARR; } LPTIM_TypeDef;
typedef struct { volatile uint32_t BRR, CR1, ISR, TDR; } USART_TypeDef;
extern GPIO_TypeDef sim_gpioa, sim_gpiob; extern RCC_TypeDef sim_rcc; extern DAC_TypeDef sim_dac; extern EXTI_TypeDef sim_exti;
extern SYSCFG_TypeDef sim_syscfg; extern PWR_TypeDef sim_pwr; extern LPTIM_TypeDef sim_lptim; extern USART_TypeDef sim_usart;
#define GPIOA (&sim_gpioa)
#define GPIOB (&sim_gpiob)
#define RCC (&sim_rcc)
#define DAC1 (&sim_dac)
#define EXTI (&sim_exti)
#define SYSCFG (&sim_syscfg)
#define PWR (&sim_pwr)
#define LPTIM1 (&sim_lptim)
#define USART2 (&sim_usart)
#define RCC_AHB2ENR_GPIOAEN (1u << 0)
#define RCC_AHB2ENR_GPIOBEN (1u << 1)
#define RCC_APB1ENR1_DAC1EN (1u << 29)
#define RCC_APB1ENR1_LPTIM1EN (1u << 31)
#define RCC_APB1ENR1_PWREN (1u << 28)
#define RCC_APB1ENR1_USART2EN (1u << 17)
#define RCC_APB2ENR_SYSCFGEN (1u << 0)
#define RCC_BDCR_LSEON (1u << 0)
#define RCC_BDCR_LSERDY (1u << 1)
#define RCC_CCIPR_LPTIM1SEL_Pos 18
#define RCC_CCIPR_LPTIM1SEL (3u << 18)
#define PWR_CR1_DBP (1u << 8)
#define DAC_CR_EN1 (1u << 0)
#define DAC_CR_EN2 (1u << 16)
#define EXTI_PR1_PIF0 (1u << 0)
#define EXTI_PR1_PIF12 (1u << 12)
#define EXTI_RTSR1_RT0 (1u << 0)
#define EXTI_RTSR1_RT12 (1u << 12)
#define EXTI_IMR1_IM0 (1u << 0)
#define EXTI_IMR1_IM12 (1u << 12)
#define SYSCFG_EXTICR1_EXTI0 (7u << 0)
#define SYSCFG_EXTICR1_EXTI0_PB (1u << 0)
#define SYSCFG_EXTICR4_EXTI12 (7u << 0)
#define SYSCFG_EXTICR4_EXTI12_PA (0u)
#define LPTIM_ISR_ARRM (1u << 1)
#define LPTIM_ICR_ARRMCF (1u << 1)
#define LPTIM_IER_ARRMIE (1u << 1)
#define LPTIM_CR_ENABLE (1u << 0)
#define LPTIM_CR_CNTSTRT (1u << 2)
#define USART_CR1_TE (1u << 3)
#define USART_CR1_UE (1u << 0)
#define USART_ISR_TXE (1u << 7)
enum { EXTI0_IRQn = 6, EXTI15_10_IRQn = 40, LPTIM1_IRQn = 65 };
#define NVIC_EnableIRQ(x) ((void)(x))
void sim_wfi(void);
void sim_settle(void);
void sim_uart_putc(char c);
#define __WFI() sim_wfi()
#endif
