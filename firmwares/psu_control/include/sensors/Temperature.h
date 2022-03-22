#ifndef TEMPERATURE_H
#define TEMPERATURE_H

#include <cstdint>

class Temperature
{
    uint8_t m_adcChannel;
    float m_ntcR;
    float m_ntcBeta;
    float m_r;

public:
    Temperature(uint8_t adcChannel, float ntcR, float ntcBeta, float r);
    float readCelsius();
};

#endif
