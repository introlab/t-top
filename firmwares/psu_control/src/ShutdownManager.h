#ifndef PSU_CONTROL_SHUTDOWN_MANAGER_H
#define PSU_CONTROL_SHUTDOWN_MANAGER_H

#include "config.h"

#include <ClassMacro.h>

#include <Arduino.h>

#include <cstdint>

class ShutdownManager
{
    uint8_t m_powerOffPin;
    uint8_t m_powerSwitchPin;

    volatile bool m_isShutdownRequested;
    bool m_isShutdownPending;
    uint32_t m_shutdownRequestHandlingTime;

public:
    ShutdownManager(uint8_t powerOffPin, uint8_t powerSwitchPin);

    DECLARE_NOT_COPYABLE(ShutdownManager);
    DECLARE_NOT_MOVABLE(ShutdownManager);

    void begin();

    bool isShutdownRequested();
    bool isShutdownPending();
    bool hasShutdownRequestTimeout();

    void setShutdownRequestHandled();
    void shutdown();
};

inline bool ShutdownManager::isShutdownRequested()
{
    return m_isShutdownRequested;
}

inline bool ShutdownManager::isShutdownPending()
{
    return m_isShutdownPending;
}

inline void ShutdownManager::setShutdownRequestHandled()
{
    m_isShutdownRequested = false;
    m_isShutdownPending = true;
    m_shutdownRequestHandlingTime = millis();
}

inline bool ShutdownManager::hasShutdownRequestTimeout()
{
    return (millis() - m_shutdownRequestHandlingTime) > SHUTDOWN_REQUEST_TIMEOUT_MS;
}

inline void ShutdownManager::shutdown()
{
    digitalWrite(m_powerOffPin, false);
}

#endif
