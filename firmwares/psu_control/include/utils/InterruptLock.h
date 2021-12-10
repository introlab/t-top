#ifndef INTERRUPT_LOCK_H
#define INTERRUPT_LOCK_H

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
