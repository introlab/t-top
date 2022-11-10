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
    SHUTDOWN = 9,
    READY_FOR_SHUTDOWN = 10
};

template<>
struct enum_min<MessageType> : std::integral_constant<uint16_t, 0>
{
};

template<>
struct enum_max<MessageType> : std::integral_constant<uint16_t, 10>
{
};


class MessageHeader
{
    static uint16_t m_messageIdCounter;

    Device m_source;
    Device m_destimation;
    bool m_acknowledgmentNeeded;
    uint16_t m_messageId;
    MessageType m_messageType;

    MessageHeader(Device source, Device destimation, bool acknowledgmentNeeded, uint16_t m_messageId, MessageType messageType);

public:
    MessageHeader(Device source, Device destimation, bool acknowledgmentNeeded, MessageType messageType);

    Device source() const;
    Device destimation() const;
    bool acknowledgmentNeeded() const;
    uint16_t messageId() const;
    MessageType messageType() const;

    template<size_t N>
    bool writeTo(SerialCommunicationBuffer<N>& buffer);
    bool writeTo(SerialCommunicationBufferView& buffer);
    static tl::optional<MessageHeader> readFrom(const SerialCommunicationBufferView& buffer);

    static void resetMessageIdCounter();
    static void setMessageIdCounter(uint16_t messageIdCounter);
};

inline Device MessageHeader::source() const
{
    return m_source;
}

inline Device MessageHeader::destimation() const
{
    return m_destimation;
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

template<size_t N>
bool MessageHeader::writeTo(SerialCommunicationBuffer<N>& buffer)
{
    SerialCommunicationBufferView view(buffer);
    return writeTo(view);
}



#endif
