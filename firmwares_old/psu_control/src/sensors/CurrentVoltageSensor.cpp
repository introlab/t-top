#include "sensors/CurrentVoltageSensor.h"

CurrentVoltageSensor::CurrentVoltageSensor(TwoWire& wire) : m_ina(wire) {}

#if INA_TYPE == INA220_TYPE
constexpr uint8_t INA220_COUNT = 1;
static uint8_t INA220_ADDRESSES[INA220_COUNT] = {INA_ADDRESS};
#endif

bool CurrentVoltageSensor::begin()
{
#if INA_TYPE == INA220_TYPE
    constexpr uint8_t max_current = static_cast<uint8_t>(INA_MAX_CURRENT);
    constexpr uint16_t shunt_resistor = static_cast<uint16_t>(INA_SHUNT_RESISTOR * 1000000);
    return m_ina.begin(
               max_current,
               shunt_resistor,
               INA_ADC_MODE_128AVG,
               INA_ADC_MODE_128AVG,
               INA_MODE_CONTINUOUS_BOTH,
               INA220_ADDRESSES,
               INA220_COUNT) == INA220_COUNT;
#elif INA_TYPE == INA226_TYPE
    if (!m_ina.begin(INA_ADDRESS))
    {
        return false;
    }
    if (m_ina.configure(
            INA226_AVERAGES_128,
            INA226_BUS_CONV_TIME_1100US,
            INA226_SHUNT_CONV_TIME_1100US,
            INA226_MODE_SHUNT_BUS_CONT))
    {
        return false;
    }
    return m_ina.calibrate(INA_SHUNT_RESISTOR, INA_MAX_CURRENT);
#endif
}

float CurrentVoltageSensor::readVoltage()
{
#if INA_TYPE == INA220_TYPE
    return m_ina.getBusMilliVolts(0) / 1000.f;
#elif INA_TYPE == INA226_TYPE
    return m_ina.readBusVoltage();
#endif
}

float CurrentVoltageSensor::readCurrent()
{
#if INA_TYPE == INA220_TYPE
    return m_ina.getBusMicroAmps(0) / 1000000.f;
#elif INA_TYPE == INA226_TYPE
    return m_ina.readShuntCurrent();
#endif
}
