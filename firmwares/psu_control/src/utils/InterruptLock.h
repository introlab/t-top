#ifndef PSU_CONTROL_UTILS_INTERRUPT_LOCK_H
#define PSU_CONTROL_UTILS_INTERRUPT_LOCK_H

class InterruptLock
{
public:
    InterruptLock();
    ~InterruptLock();
};

class PinInterruptLock
{
public:
    PinInterruptLock();
    ~PinInterruptLock();
};

#endif
