#ifndef PSU_CONTROL_COMMAND_SENDER_H
#define PSU_CONTROL_COMMAND_SENDER_H

#include <cstdint>

class PsuControlCommandSender {
  uint8_t m_buffer[UINT8_MAX];

public:
  PsuControlCommandSender();

  void sendVolumeCommand(uint8_t volume);

private:
  void sendCommand();
};

#endif
