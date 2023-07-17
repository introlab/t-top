#include "Fan.h"
#include "../config.h"

#include <Arduino.h>

constexpr int HALF_SPEED = static_cast<int>(PWM_MAX_VALUE / 2);
constexpr int FULL_SPEED = static_cast<int>(PWM_MAX_VALUE);

Fan::Fan(uint8_t pin) : m_pin(pin), m_speed(0) {}

void Fan::begin()
{
    pinMode(m_pin, OUTPUT);
    analogWriteFrequency(m_pin, FAN_PWM_FREQUENCY);

    analogWrite(m_pin, 0);
    m_speed = 0;
}

void Fan::update(float celcius)
{
    if (celcius <= (FAN_TEMPERATURE_STEP_1 - FAN_HYSTERESIS) && m_speed >= HALF_SPEED)
    {
        analogWrite(m_pin, 0);
        m_speed = 0;
    }
    else if (celcius <= (FAN_TEMPERATURE_STEP_2 - FAN_HYSTERESIS) && m_speed >= FULL_SPEED)
    {
        analogWrite(m_pin, HALF_SPEED);
        m_speed = HALF_SPEED;
    }
    else if (celcius >= (FAN_TEMPERATURE_STEP_2 + FAN_HYSTERESIS) && m_speed <= HALF_SPEED)
    {
        analogWrite(m_pin, FULL_SPEED);
        m_speed = FULL_SPEED;
    }
    else if (celcius >= (FAN_TEMPERATURE_STEP_1 + FAN_HYSTERESIS) && m_speed <= 0)
    {
        analogWrite(m_pin, HALF_SPEED);
        m_speed = HALF_SPEED;
    }
}
