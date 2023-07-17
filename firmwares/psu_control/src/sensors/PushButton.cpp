#include "PushButton.h"

#include <Arduino.h>

#include <cstddef>

constexpr size_t MAXIMUM_BUTTON_COUNT = 4;
static volatile bool* interruptDetected[MAXIMUM_BUTTON_COUNT] = {nullptr};
static size_t interruptDetectedIndex = 0;

static void buttonInterrupt0()
{
    *interruptDetected[0] = true;
}

static void buttonInterrupt1()
{
    *interruptDetected[1] = true;
}

static void buttonInterrupt2()
{
    *interruptDetected[2] = true;
}

static void buttonInterrupt3()
{
    *interruptDetected[3] = true;
}

PushButton::PushButton(uint8_t pin, PushButtonType type) : m_pin(pin), m_type(type) {}

bool PushButton::begin()
{
    if (interruptDetectedIndex >= MAXIMUM_BUTTON_COUNT)
    {
        return false;
    }

    interruptDetected[interruptDetectedIndex] = &m_interruptDetected;

    pinMode(m_pin, INPUT);
    if (interruptDetectedIndex == 0)
    {
        attachInterrupt(digitalPinToInterrupt(m_pin), buttonInterrupt0, RISING);
    }
    else if (interruptDetectedIndex == 1)
    {
        attachInterrupt(digitalPinToInterrupt(m_pin), buttonInterrupt1, RISING);
    }
    else if (interruptDetectedIndex == 2)
    {
        attachInterrupt(digitalPinToInterrupt(m_pin), buttonInterrupt2, RISING);
    }
    else if (interruptDetectedIndex == 3)
    {
        attachInterrupt(digitalPinToInterrupt(m_pin), buttonInterrupt3, RISING);
    }

    interruptDetectedIndex++;
    return true;
}

bool PushButton::read()
{
    bool value = false;

    if (m_type == PushButtonType::SINGLE)
    {
        value = m_interruptDetected;
    }
    else if (m_type == PushButtonType::REPEATABLE)
    {
        value = m_interruptDetected || digitalRead(m_pin);
    }

    m_interruptDetected = false;
    return value;
}
