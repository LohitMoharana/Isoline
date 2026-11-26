# Build: make CMSIS_DEVICE=<path to cmsis_device_l4> CMSIS_CORE=<path to CMSIS/Core/Include>
# (git clone https://github.com/STMicroelectronics/cmsis_device_l4 and https://github.com/STMicroelectronics/cmsis_core)
CMSIS_DEVICE ?= ../../vendor/cmsis_device_l4
CMSIS_CORE ?= ../../vendor/cmsis_core/CMSIS/Core/Include
CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -std=c99 -Wall -ffunction-sections -fdata-sections \
         -DSTM32L432xx -I. -I../core -I../../export -I$(CMSIS_DEVICE)/Include -I$(CMSIS_CORE)
LDFLAGS = -T STM32L432KC.ld -Wl,--gc-sections -specs=nano.specs -specs=nosys.specs -lm -Wl,-Map=isoline.map
SRC = main.c ../core/isoline_core.c ../core/isoline_stream.c $(CMSIS_DEVICE)/Source/Templates/system_stm32l4xx.c \
      $(CMSIS_DEVICE)/Source/Templates/gcc/startup_stm32l432xx.s
isoline.elf: $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $@
	arm-none-eabi-size $@
	arm-none-eabi-objcopy -O binary $@ isoline.bin
clean:
	rm -f isoline.elf isoline.bin isoline.map
