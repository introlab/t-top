#include "SerialMessages.h"

constexpr uint8_t MessageHeader::HEADER_SIZE;

uint16_t MessageHeader::m_messageIdCounter(0);

MessageHeader::MessageHeader(
    Device source,
    Device destination,
    bool acknowledgmentNeeded,
    uint16_t messageId,
    MessageType messageType)
    : m_source(source),
      m_destination(destination),
      m_acknowledgmentNeeded(acknowledgmentNeeded),
      m_messageId(messageId),
      m_messageType(messageType)
{
}

MessageHeader::MessageHeader(Device source, Device destination, bool acknowledgmentNeeded, MessageType messageType)
    : m_source(source),
      m_destination(destination),
      m_acknowledgmentNeeded(acknowledgmentNeeded),
      m_messageId(m_messageIdCounter++),
      m_messageType(messageType)
{
}
