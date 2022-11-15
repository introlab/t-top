#ifndef PSU_CONTROL_SENSORS_CURRENT_VOLTAGE_SENSOR_H
#define PSU_CONTROL_SENSORS_CURRENT_VOLTAGE_SENSOR_H

#include <ClassMacro.h>

#include <Wire.h>

#include <INA220.h>

class CurrentVoltageSensor
{
    INA220 m_ina;
    uint8_t m_ina220Address;
    float m_shuntResistor;
    uint8_t m_maxCurrent;

public:
    CurrentVoltageSensor(TwoWire& wire, uint8_t ina220Address, float shuntResistor, uint8_t maxCurrent);

    DECLARE_NOT_COPYABLE(CurrentVoltageSensor);
    DECLARE_NOT_MOVABLE(CurrentVoltageSensor);

    bool begin();

    float readVoltage();
    float readCurrent();
};

#endif
