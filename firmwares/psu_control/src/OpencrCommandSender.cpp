#include "OpencrCommandSender.h"
#include "config.h"

#include <Arduino.h>

constexpr uint8_t MESSAGE_TYPE_STATUS = 0;

OpencrCommandSender::OpencrCommandSender()
{
    memset(m_buffer, 0, UINT8_MAX);
}

void OpencrCommandSender::sendStatusCommand(bool isBatteryCharging, float stateOfCharge, float current, float voltage)
{
    m_buffer[0] = sizeof(float) * 3 + 3;
    m_buffer[1] = MESSAGE_TYPE_STATUS;
    m_buffer[2] = static_cast<uint8_t>(isBatteryCharging);

    memcpy(m_buffer + 3, &stateOfCharge, sizeof(float));
    memcpy(m_buffer + 7, &current, sizeof(float));
    memcpy(m_buffer + 11, &voltage, sizeof(float));

    sendCommand();
}

void OpencrCommandSender::sendCommand()
{
    for (uint8_t i = 0; i < m_buffer[0]; i++)
    {
        OPENCR_SERIAL.write(m_buffer[i]);
    }

    memset(m_buffer, 0, UINT8_MAX);
}
