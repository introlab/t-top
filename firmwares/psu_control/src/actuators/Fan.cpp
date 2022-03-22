#include "actuators/Fan.h"
#include "config.h"

#include <Arduino.h>

constexpr int HALF_SPEED = static_cast<int>(PWM_MAX_VALUE / 2);
constexpr int FULL_SPEED = static_cast<int>(PWM_MAX_VALUE);

Fan::Fan() : m_speed(0) {}

void Fan::begin()
{
    pinMode(FAN_PIN, OUTPUT);
    analogWriteFrequency(FAN_PIN, FAN_PWM_FREQUENCY);

    analogWrite(FAN_PIN, 0);
    m_speed = 0;
}

void Fan::update(float celcius)
{
    if (celcius <= (FAN_TEMPERATURE_STEP_1 - FAN_HYSTERESIS) && m_speed >= HALF_SPEED)
    {
        analogWrite(FAN_PIN, 0);
        m_speed = 0;
    }
    else if (celcius <= (FAN_TEMPERATURE_STEP_2 - FAN_HYSTERESIS) && m_speed >= FULL_SPEED)
    {
        analogWrite(FAN_PIN, HALF_SPEED);
        m_speed = HALF_SPEED;
    }
    else if (celcius >= (FAN_TEMPERATURE_STEP_2 + FAN_HYSTERESIS) && m_speed <= HALF_SPEED)
    {
        analogWrite(FAN_PIN, FULL_SPEED);
        m_speed = FULL_SPEED;
    }
    else if (celcius >= (FAN_TEMPERATURE_STEP_1 + FAN_HYSTERESIS) && m_speed <= 0)
    {
        analogWrite(FAN_PIN, HALF_SPEED);
        m_speed = HALF_SPEED;
    }
}
