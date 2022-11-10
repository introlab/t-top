#ifndef OPENCR_COMMAND_HANDLER_H
#define OPENCR_COMMAND_HANDLER_H

#include <cstdint>

typedef void (*VolumeCommandHandler)(uint8_t);

class OpencrCommandHandler
{
    VolumeCommandHandler m_volumeCommandHandler;

    uint8_t m_buffer[UINT8_MAX];
    uint8_t m_bufferIndex;

public:
    OpencrCommandHandler();
    void update();

    void setVolumeCommandHandler(VolumeCommandHandler handler);

private:
    void readOneByte();
    void handleMessage();
    void clearMessage();
};

#endif
