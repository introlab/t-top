#ifndef PSU_CONTROL_ACTUATORS_BUZZER_H
#define PSU_CONTROL_ACTUATORS_BUZZER_H

#include <ClassMacro.h>

#include <Arduino.h>

class Buzzer
{
    uint8_t m_pin;

    volatile bool m_enabled;
    volatile int m_value;
    IntervalTimer m_timer;

public:
    explicit Buzzer(uint8_t pin);

    DECLARE_NOT_COPYABLE(Buzzer);
    DECLARE_NOT_MOVABLE(Buzzer);

    bool begin();

    void enable();
    void disable();
};

#endif
