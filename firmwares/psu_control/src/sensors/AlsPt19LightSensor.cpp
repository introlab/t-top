#include "AlsPt19LightSensor.h"
#include "../config.h"

#include <Arduino.h>
#include <cmath>

using namespace std;

AlsPt19LightSensor::AlsPt19LightSensor(uint8_t pin) : m_pin(pin) {}

void AlsPt19LightSensor::begin()
{
    pinMode(m_pin, INPUT);
}

float AlsPt19LightSensor::read()
{
    float adcValue = analogRead(m_pin);
    float clampedAdcValue = max(LIGHT_SENSOR_MINIMUM_VALUE, min(adcValue, LIGHT_SENSOR_MAXIMUM_VALUE));
    return (clampedAdcValue - LIGHT_SENSOR_MINIMUM_VALUE) / LIGHT_SENSOR_MAXIMUM_VALUE;
}
