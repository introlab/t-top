#ifndef PSU_CONTROL_SENSORS_BUTTON_H
#define PSU_CONTROL_SENSORS_BUTTON_H

#include <ClassMacro.h>

#include <cstdint>

enum class PushButtonType
{
    SINGLE,
    REPEATABLE
};

class PushButton
{
    uint8_t m_pin;
    PushButtonType m_type;
    volatile bool m_interruptDetected;

public:
    PushButton(uint8_t pin, PushButtonType type);

    DECLARE_NOT_COPYABLE(PushButton);
    DECLARE_NOT_MOVABLE(PushButton);

    bool begin();
    bool read();
};

#endif
