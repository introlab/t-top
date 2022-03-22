#ifndef PSU_CONTROL_COMMAND_HANDLER_H
#define PSU_CONTROL_COMMAND_HANDLER_H

#include <cstdint>

typedef void (*StatusCommandHandler)(bool isBatteryCharging, float stateOfCharge, float current, float voltage);

class PsuControlCommandHandler
{
    StatusCommandHandler m_statusCommandHandler;

    uint8_t m_buffer[UINT8_MAX];
    uint8_t m_bufferIndex;

public:
    PsuControlCommandHandler();
    void update();

    void setStatusCommandHandler(StatusCommandHandler handler);

private:
    void readOneByte();
    void handleMessage();
    void clearMessage();
};

#endif
