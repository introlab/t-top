#include "PsuControlCommandSender.h"

#include <Arduino.h>

constexpr uint8_t MESSAGE_TYPE_VOLUME = 1;

PsuControlCommandSender::PsuControlCommandSender() {
  memset(m_buffer, 0, UINT8_MAX);
}

void PsuControlCommandSender::sendVolumeCommand(uint8_t volume) {
  m_buffer[0] = 3;
  m_buffer[1] = MESSAGE_TYPE_VOLUME;
  m_buffer[2] = volume;

  sendCommand();
}

void PsuControlCommandSender::sendCommand() {
  for (uint8_t i = 0; i < m_buffer[0]; i++) {
    Serial1.write(m_buffer[i]);
  }

  memset(m_buffer, 0, UINT8_MAX);
}
