#include "CurrentVoltageSensor.h"

CurrentVoltageSensor::CurrentVoltageSensor(
    TwoWire& wire,
    uint8_t ina220Address,
    float shuntResistor,
    uint8_t maxCurrent)
    : m_ina(wire),
      m_ina220Address(ina220Address),
      m_shuntResistor(shuntResistor),
      m_maxCurrent(maxCurrent)
{
}

bool CurrentVoltageSensor::begin()
{
    return m_ina.begin(
               m_maxCurrent,
               static_cast<uint16_t>(m_shuntResistor * 1000000),
               INA_ADC_MODE_128AVG,
               INA_ADC_MODE_128AVG,
               INA_MODE_CONTINUOUS_BOTH,
               &m_ina220Address,
               1) == 1;
}

float CurrentVoltageSensor::readVoltage()
{
    return m_ina.getBusMilliVolts(0) / 1000.f;
}

float CurrentVoltageSensor::readCurrent()
{
    return m_ina.getBusMicroAmps(0) / 1000000.f;
}
