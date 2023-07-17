#ifndef PSU_CONTROL_ACTUATORS_FAN_H
#define PSU_CONTROL_ACTUATORS_FAN_H

#include <ClassMacro.h>

#include <cstdint>

class Fan
{
    uint8_t m_pin;
    int m_speed;

public:
    explicit Fan(uint8_t pin);

    DECLARE_NOT_COPYABLE(Fan);
    DECLARE_NOT_MOVABLE(Fan);

    void begin();

    void update(float celcius);
};

#endif
