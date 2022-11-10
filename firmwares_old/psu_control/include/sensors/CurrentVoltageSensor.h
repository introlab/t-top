#ifndef CURRENT_VOLTAGE_SENSOR_H
#define CURRENT_VOLTAGE_SENSOR_H

#include "config.h"

#include <Wire.h>

#if INA_TYPE == INA220_TYPE
#include <INA220.h>
#elif INA_TYPE == INA226_TYPE
#include <INA226.h>
#endif

class CurrentVoltageSensor
{
#if INA_TYPE == INA220_TYPE
    INA220 m_ina;
#elif INA_TYPE == INA226_TYPE
    INA226 m_ina;
#endif

public:
    CurrentVoltageSensor(TwoWire& wire);
    bool begin();

    float readVoltage();
    float readCurrent();
};

#endif
