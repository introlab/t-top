#ifndef OPENCR_COMMAND_SENDER_H
#define OPENCR_COMMAND_SENDER_H

#include <cstdint>

class OpencrCommandSender
{
    uint8_t m_buffer[UINT8_MAX];

public:
    OpencrCommandSender();

    void sendStatusCommand(
        bool isPsuConnected,
        bool isBatteryCharging,
        float stateOfCharge,
        float current,
        float voltage);

private:
    void sendCommand();
};

#endif
