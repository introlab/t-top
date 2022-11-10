#include "utils/InterruptLock.h"

#include <Arduino.h>

InterruptLock::InterruptLock()
{
    noInterrupts();
}

InterruptLock::~InterruptLock()
{
    interrupts();
}

PinInterruptLock::PinInterruptLock()
{
    NVIC_DISABLE_IRQ(IRQ_PORTA);
    NVIC_DISABLE_IRQ(IRQ_PORTCD);
}

PinInterruptLock::~PinInterruptLock()
{
    NVIC_ENABLE_IRQ(IRQ_PORTA);
    NVIC_ENABLE_IRQ(IRQ_PORTCD);
}
