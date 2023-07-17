#ifndef PSU_CONTROL_SENSORS_THERMISTOR_H
#define PSU_CONTROL_SENSORS_THERMISTOR_H

#include <ClassMacro.h>

#include <cstdint>

class Thermistor
{
    uint8_t m_pin;
    float m_ntcR;
    float m_ntcBeta;
    float m_r;

public:
    Thermistor(uint8_t pin, float ntcR, float ntcBeta, float r);

    DECLARE_NOT_COPYABLE(Thermistor);
    DECLARE_NOT_MOVABLE(Thermistor);

    void begin();

    float readCelsius();
};

#endif
