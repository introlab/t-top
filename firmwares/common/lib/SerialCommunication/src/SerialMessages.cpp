#include "SerialMessages.h"


#define CHECK_BUFFER_PUT(code) if (!(code)) { return false; }

#define CHECK_BUFFER_READ(result, code) \
    do \
    { \
        auto tmp = (code); \
        if (tmp.has_value()) \
        { \
            (result) = tmp.value(); \
        } \
        else \
        { \
            return tl::nullopt; \
        } \
    } while (false)


uint16_t MessageHeader::m_messageIdCounter(0);

MessageHeader::MessageHeader(Device source, Device destimation, bool acknowledgmentNeeded, uint16_t messageId, MessageType messageType) :
    m_source(source),
    m_destimation(destimation),
    m_acknowledgmentNeeded(acknowledgmentNeeded),
    m_messageId(messageId),
    m_messageType(messageType)
{
}

MessageHeader::MessageHeader(Device source, Device destimation, bool acknowledgmentNeeded, MessageType messageType) :
    m_source(source),
    m_destimation(destimation),
    m_acknowledgmentNeeded(acknowledgmentNeeded),
    m_messageId(m_messageIdCounter++),
    m_messageType(messageType)
{
}

bool MessageHeader::writeTo(SerialCommunicationBufferView& buffer)
{
    CHECK_BUFFER_PUT(buffer.put(m_source));
    CHECK_BUFFER_PUT(buffer.put(m_destimation));
    CHECK_BUFFER_PUT(buffer.put(m_acknowledgmentNeeded));
    CHECK_BUFFER_PUT(buffer.put(m_messageId));
    CHECK_BUFFER_PUT(buffer.put(m_messageType));

    return true;
}

tl::optional<MessageHeader> MessageHeader::readFrom(const SerialCommunicationBufferView& buffer)
{
    Device source;
    CHECK_BUFFER_READ(source, buffer.read<Device>());

    Device destimation;
    CHECK_BUFFER_READ(destimation, buffer.read<Device>());

    bool acknowledgmentNeeded;
    CHECK_BUFFER_READ(acknowledgmentNeeded, buffer.read<bool>());

    uint16_t messageId;
    CHECK_BUFFER_READ(messageId, buffer.read<uint16_t>());

    MessageType messageType;
    CHECK_BUFFER_READ(messageType, buffer.read<MessageType>());

    return MessageHeader(source, destimation, acknowledgmentNeeded, messageId, messageType);
}
