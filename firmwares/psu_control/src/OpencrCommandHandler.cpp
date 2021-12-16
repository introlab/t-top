#include "OpencrCommandHandler.h"
#include "config.h"

#include <Arduino.h>

constexpr uint8_t MESSAGE_SIZE_INDEX = 0;
constexpr uint8_t MESSAGE_TYPE_INDEX = 1;

constexpr uint8_t MESSAGE_TYPE_VOLUME = 1;

OpencrCommandHandler::OpencrCommandHandler() : m_volumeCommandHandler(nullptr) {
  clearMessage();
}

void OpencrCommandHandler::update() {
  while (OPENCR_SERIAL.available() > 0) {
    readOneByte();
  }
}

void OpencrCommandHandler::setVolumeCommandHandler(VolumeCommandHandler handler) {
    m_volumeCommandHandler = handler;
}

void OpencrCommandHandler::readOneByte() {
  m_buffer[m_bufferIndex] = OPENCR_SERIAL.read();
  m_bufferIndex++;

  if (m_bufferIndex >= m_buffer[MESSAGE_SIZE_INDEX]) {
    handleMessage();
  }
}

void OpencrCommandHandler::handleMessage() {
  switch (m_buffer[MESSAGE_TYPE_INDEX]) {
    case MESSAGE_TYPE_VOLUME:
      if (m_volumeCommandHandler != nullptr) {
        m_volumeCommandHandler(m_buffer[2]);
      }
      break;
  }

  clearMessage();
}

void OpencrCommandHandler::clearMessage() {
  memset(m_buffer, 0, UINT8_MAX);
  m_bufferIndex = 0;
}
