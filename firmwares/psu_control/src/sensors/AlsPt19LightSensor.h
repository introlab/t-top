#ifndef PSU_CONTROL_SENSORS_ALS_PT19_LIGHT_SENSOR_H
#define PSU_CONTROL_SENSORS_ALS_PT19_LIGHT_SENSOR_H

#include <ClassMacro.h>

#include <cstdint>

class AlsPt19LightSensor
{
    uint8_t m_pin;

public:
    explicit AlsPt19LightSensor(uint8_t pin);

    DECLARE_NOT_COPYABLE(AlsPt19LightSensor);
    DECLARE_NOT_MOVABLE(AlsPt19LightSensor);

    void begin();
    float read();  // 0 to 1
};

#endif
