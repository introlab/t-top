#ifndef SERIAL_COMMUNICATION_SERIAL_MESSAGES_H
#define SERIAL_COMMUNICATION_SERIAL_MESSAGES_H

#include "SerialCommunicationBuffer.h"

#include <tl/optional.hpp>

#include <cstdint>

enum class Device : uint8_t
{
    PSU_CONTROL = 0,
    DYNAMIXEL_CONTROL = 1,
    COMPUTER = 2
};

template<>
struct enum_min<Device> : std::integral_constant<uint8_t, 0>
{
};

template<>
struct enum_max<Device> : std::integral_constant<uint8_t, 2>
{
};


enum class MessageType : uint16_t
{
    ACKNOWLEDGMENT = 0,
    BASE_STATUS = 1,
    BUTTON_PRESSED = 2,
    SET_VOLUME = 3,
    SET_LED_COLORS = 4,
    MOTOR_STATUS = 5,
    IMU_DATA = 6,
    SET_TORSO_ORIENTATION = 7,
    SET_HEAD_POSE = 8,
    SHUTDOWN = 9
};

template<>
struct enum_min<MessageType> : std::integral_constant<uint16_t, 0>
{
};

template<>
struct enum_max<MessageType> : std::integral_constant<uint16_t, 10>
{
};


#define CHECK_BUFFER_WRITE(code)                                                                                       \
    if (!(code))                                                                                                       \
    {                                                                                                                  \
        return false;                                                                                                  \
    }

#define CHECK_BUFFER_READ(result, code)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        auto tmp = (code);                                                                                             \
        if (tmp.has_value())                                                                                           \
        {                                                                                                              \
            (result) = *tmp;                                                                                           \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            return tl::nullopt;                                                                                        \
        }                                                                                                              \
    } while (false)


class MessageHeader
{
public:
    static constexpr uint8_t HEADER_SIZE = 7;

private:
    static uint16_t m_messageIdCounter;

    Device m_source;
    Device m_destination;
    bool m_acknowledgmentNeeded;
    uint16_t m_messageId;
    MessageType m_messageType;

    MessageHeader(
        Device source,
        Device destination,
        bool acknowledgmentNeeded,
        uint16_t m_messageId,
        MessageType messageType);

public:
    MessageHeader(Device source, Device destination, bool acknowledgmentNeeded, MessageType messageType);

    Device source() const;
    Device destination() const;
    bool acknowledgmentNeeded() const;
    uint16_t messageId() const;
    MessageType messageType() const;

    template<class Buffer>
    bool writeTo(Buffer& buffer) const;

    template<class Buffer>
    static tl::optional<MessageHeader> readFrom(Buffer& buffer);

    static void resetMessageIdCounter();
    static void setMessageIdCounter(uint16_t messageIdCounter);
};

inline Device MessageHeader::source() const
{
    return m_source;
}

inline Device MessageHeader::destination() const
{
    return m_destination;
}

inline bool MessageHeader::acknowledgmentNeeded() const
{
    return m_acknowledgmentNeeded;
}

inline uint16_t MessageHeader::messageId() const
{
    return m_messageId;
}

inline MessageType MessageHeader::messageType() const
{
    return m_messageType;
}

inline void MessageHeader::resetMessageIdCounter()
{
    m_messageIdCounter = 0;
}

inline void MessageHeader::setMessageIdCounter(uint16_t messageIdCounter)
{
    m_messageIdCounter = messageIdCounter;
}

template<class Buffer>
bool MessageHeader::writeTo(Buffer& buffer) const
{
    CHECK_BUFFER_WRITE(buffer.write(m_source));
    CHECK_BUFFER_WRITE(buffer.write(m_destination));
    CHECK_BUFFER_WRITE(buffer.write(m_acknowledgmentNeeded));
    CHECK_BUFFER_WRITE(buffer.write(m_messageId));
    CHECK_BUFFER_WRITE(buffer.write(m_messageType));

    return true;
}

template<class Buffer>
tl::optional<MessageHeader> MessageHeader::readFrom(Buffer& buffer)
{
    Device source;
    Device destination;
    bool acknowledgmentNeeded;
    uint16_t messageId;
    MessageType messageType;

    CHECK_BUFFER_READ(source, buffer.template read<Device>());
    CHECK_BUFFER_READ(destination, buffer.template read<Device>());
    CHECK_BUFFER_READ(acknowledgmentNeeded, buffer.template read<bool>());
    CHECK_BUFFER_READ(messageId, buffer.template read<uint16_t>());
    CHECK_BUFFER_READ(messageType, buffer.template read<MessageType>());

    return MessageHeader(source, destination, acknowledgmentNeeded, messageId, messageType);
}


class AcknowledgmentPayload;

template<class Payload>
class Message
{
    MessageHeader m_header;
    Payload m_payload;

public:
    Message(Device source, Device destination, Payload payload);
    Message(Device source, Device destination, bool acknowledgmentNeeded, Payload payload);

    const MessageHeader& header() const;
    const Payload& payload() const;
};

template<class Payload>
Message<Payload>::Message(Device source, Device destination, Payload payload)
    : m_header(source, destination, Payload::DEFAULT_ACKNOWLEDGMENT_NEEDED, Payload::MESSAGE_TYPE),
      m_payload(payload)
{
}

template<class Payload>
Message<Payload>::Message(Device source, Device destination, bool acknowledgmentNeeded, Payload payload)
    : m_header(source, destination, acknowledgmentNeeded, Payload::MESSAGE_TYPE),
      m_payload(payload)
{
    static_assert(
        !std::is_same<Payload, AcknowledgmentPayload>::value,
        "AcknowledgmentNeeded cannot be set for AcknowledgmentPayload");
}

template<class Payload>
inline const MessageHeader& Message<Payload>::header() const
{
    return m_header;
}

template<class Payload>
inline const Payload& Message<Payload>::payload() const
{
    return m_payload;
}

#endif
