#include "Buzzer.h"
#include "../config.h"
#include "../utils/InterruptLock.h"

constexpr int ON_PWM_VALUE = static_cast<int>(PWM_MAX_VALUE / 2);


static volatile uint8_t pin;
static volatile bool* enabled;
static volatile int* value;
static void buzzerTimerInterrupt()
{
    if (*value == 0 && *enabled)
    {
        analogWrite(pin, ON_PWM_VALUE);
        *value = ON_PWM_VALUE;
    }
    else
    {
        analogWrite(pin, 0);
        *value = 0;
    }
}

Buzzer::Buzzer(uint8_t pin) : m_pin(pin), m_enabled(false), m_value(0) {}

bool Buzzer::begin()
{
    pin = m_pin;
    enabled = &m_enabled;
    value = &m_value;

    pinMode(m_pin, OUTPUT);
    analogWriteFrequency(m_pin, BUZZER_PWM_FREQUENCY);
    disable();

    return m_timer.begin(buzzerTimerInterrupt, BUZZER_ON_OFF_INTERVAL_US);
}

void Buzzer::enable()
{
    InterruptLock lock;

    m_enabled = true;
}

void Buzzer::disable()
{
    InterruptLock lock;

    m_enabled = false;
}
