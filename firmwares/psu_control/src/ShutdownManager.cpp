#include "ShutdownManager.h"

volatile bool* isShutdownRequested;
static void powerSwitchInterrupt()
{
    *isShutdownRequested = true;
}

ShutdownManager::ShutdownManager(uint8_t powerOffPin, uint8_t powerSwitchPin)
    : m_powerOffPin(powerOffPin),
      m_powerSwitchPin(powerSwitchPin),
      m_isShutdownRequested(false),
      m_isShutdownPending(false),
      m_shutdownRequestHandlingTime(0)
{
}

void ShutdownManager::begin()
{
    pinMode(m_powerOffPin, OUTPUT);
    digitalWrite(m_powerOffPin, true);

    ::isShutdownRequested = &m_isShutdownRequested;
    pinMode(m_powerSwitchPin, INPUT);
    attachInterrupt(digitalPinToInterrupt(m_powerSwitchPin), powerSwitchInterrupt, FALLING);
}
