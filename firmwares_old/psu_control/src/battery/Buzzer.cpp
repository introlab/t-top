#include "battery/Buzzer.h"
#include "config.h"
#include "utils/InterruptLock.h"

constexpr int ON_PWM_VALUE = static_cast<int>(PWM_MAX_VALUE / 2);


static volatile bool* enabled;
static volatile int* value;
static void buzzerTimerInterrupt()
{
    if (*value == 0 && *enabled)
    {
        analogWrite(BUZZER_PIN, ON_PWM_VALUE);
        *value = ON_PWM_VALUE;
    }
    else
    {
        analogWrite(BUZZER_PIN, 0);
        *value = 0;
    }
}

Buzzer::Buzzer() : m_enabled(false), m_value(0) {}

bool Buzzer::begin()
{
    enabled = &m_enabled;
    value = &m_value;

    pinMode(BUZZER_PIN, OUTPUT);
    analogWriteFrequency(BUZZER_PIN, BUZZER_PWM_FREQUENCY);
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
