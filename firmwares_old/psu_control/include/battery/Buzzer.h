#ifndef BUZZER_H
#define BUZZER_H

#include <Arduino.h>

class Buzzer
{
    volatile bool m_enabled;
    volatile int m_value;
    IntervalTimer m_timer;

public:
    Buzzer();
    bool begin();

    void enable();
    void disable();
};

#endif
