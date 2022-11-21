#ifndef COMMUNICATION_SERIAL_COMMUNICATION_MANAGER_H
#define COMMUNICATION_SERIAL_COMMUNICATION_MANAGER_H

#include "SerialMessages.h"
#include "SerialMessagePayloads.h"

#include <ClassMacro.h>
#include <Crc8.h>

#include <tuple>
#include <limits>

#ifdef SERIAL_COMMUNICATION_MANAGER_USE_STD_FUNCTION

#include <functional>

typedef std::function<void(Device, const BaseStatusPayload&)> BaseStatusHandler;
typedef std::function<void(Device, const ButtonPressedPayload&)> ButtonPressedHandler;
typedef std::function<void(Device, const SetVolumePayload&)> SetVolumeHandler;
typedef std::function<void(Device, const SetLedColorsPayload&)> SetLedColorsHandler;
typedef std::function<void(Device, const MotorStatusPayload&)> MotorStatusHandler;
typedef std::function<void(Device, const ImuDataPayload&)> ImuDataHandler;
typedef std::function<void(Device, const SetTorsoOrientationPayload&)> SetTorsoOrientationHandler;
typedef std::function<void(Device, const SetHeadPosePayload&)> SetHeadPoseHandler;
typedef std::function<void(Device, const ShutdownPayload&)> ShutdownHandler;
typedef std::function<void(Device, const uint8_t*, size_t)> RouteCallback;
typedef std::function<void(const char*, tl::optional<MessageType>)> ErrorCallback;

#else

typedef void (*BaseStatusHandler)(Device source, const BaseStatusPayload& payload);
typedef void (*ButtonPressedHandler)(Device source, const ButtonPressedPayload& payload);
typedef void (*SetVolumeHandler)(Device source, const SetVolumePayload& payload);
typedef void (*SetLedColorsHandler)(Device source, const SetLedColorsPayload& payload);
typedef void (*MotorStatusHandler)(Device source, const MotorStatusPayload& payload);
typedef void (*ImuDataHandler)(Device source, const ImuDataPayload& payload);
typedef void (*SetTorsoOrientationHandler)(Device source, const SetTorsoOrientationPayload& payload);
typedef void (*SetHeadPoseHandler)(Device source, const SetHeadPosePayload& payload);
typedef void (*ShutdownHandler)(Device source, const ShutdownPayload& payload);

typedef void (*RouteCallback)(Device destination, const uint8_t* data, size_t size);
typedef void (*ErrorCallback)(const char* message, tl::optional<MessageType> messageType);

#endif


template<class Payload>
struct PendingMessage
{
    Message<Payload> message;
    uint32_t timestampMs;
    size_t trialCount;
};

class PendingMessages
{
    std::tuple<
        tl::optional<PendingMessage<BaseStatusPayload>>,
        tl::optional<PendingMessage<ButtonPressedPayload>>,
        tl::optional<PendingMessage<SetVolumePayload>>,
        tl::optional<PendingMessage<SetLedColorsPayload>>,
        tl::optional<PendingMessage<MotorStatusPayload>>,
        tl::optional<PendingMessage<ImuDataPayload>>,
        tl::optional<PendingMessage<SetTorsoOrientationPayload>>,
        tl::optional<PendingMessage<SetHeadPosePayload>>,
        tl::optional<PendingMessage<ShutdownPayload>>>
        m_messages;

public:
    PendingMessages();

    template<class Payload>
    void set(const Message<Payload>& message, uint32_t timestampMs);
    void clear(uint16_t messageId);

    template<class Functor, size_t I = 0>
        typename std::enable_if < I<std::tuple_size<decltype(m_messages)>::value, void>::type forEach(Functor f)
    {
        f(std::get<I>(m_messages));
        forEach<Functor, I + 1>(f);
    }

    template<class Functor, size_t I = 0>
    typename std::enable_if<I >= std::tuple_size<decltype(m_messages)>::value, void>::type forEach(Functor f)
    {
    }
};

template<class Payload>
void PendingMessages::set(const Message<Payload>& message, uint32_t timestampMs)
{
    std::get<tl::optional<PendingMessage<Payload>>>(m_messages) = PendingMessage<Payload>{message, timestampMs, 1};
}

class SerialPort
{
public:
    virtual void read(SerialCommunicationBufferView& buffer) = 0;
    virtual void write(const uint8_t* data, size_t size) = 0;
};

constexpr uint8_t SERIAL_COMMUNICATION_PREAMBLE_BYTE = 0xAA;
constexpr size_t SERIAL_COMMUNICATION_PREAMBLE_SIZE = 4;
constexpr uint8_t SERIAL_COMMUNICATION_PREAMBLE[SERIAL_COMMUNICATION_PREAMBLE_SIZE] = {
    SERIAL_COMMUNICATION_PREAMBLE_BYTE,
    SERIAL_COMMUNICATION_PREAMBLE_BYTE,
    SERIAL_COMMUNICATION_PREAMBLE_BYTE,
    SERIAL_COMMUNICATION_PREAMBLE_BYTE};

constexpr size_t SERIAL_COMMUNICATION_BUFFER_SIZE =
    std::numeric_limits<uint8_t>::max() + SERIAL_COMMUNICATION_PREAMBLE_SIZE;
constexpr uint8_t SERIAL_COMMUNICATION_MESSAGE_SIZE_SIZE = sizeof(uint8_t);
constexpr uint8_t SERIAL_COMMUNICATION_CRC8_SIZE = sizeof(uint8_t);
constexpr uint32_t SERIAL_COMMUNICATION_MAXIMUM_PAYLOAD_SIZE =
    std::numeric_limits<uint8_t>::max() - sizeof(MessageHeader) - SERIAL_COMMUNICATION_MESSAGE_SIZE_SIZE -
    SERIAL_COMMUNICATION_CRC8_SIZE;  // 2: CRC8 + message size


class SerialCommunicationManager
{
    Device m_device;
    uint32_t m_acknowledgmentTimeoutMs;
    size_t m_maximumTrialCount;
    SerialPort& m_serialPort;

    bool m_isWaitingRxPreamble;
    tl::optional<uint8_t> m_currentRxMessageSize;
    SerialCommunicationBuffer<SERIAL_COMMUNICATION_BUFFER_SIZE> m_rxBuffer;
    SerialCommunicationBuffer<SERIAL_COMMUNICATION_BUFFER_SIZE> m_txBuffer;

    BaseStatusHandler m_baseStatusHandler;
    ButtonPressedHandler m_buttonPressedHandler;
    SetVolumeHandler m_setVolumeHandler;
    SetLedColorsHandler m_setLedColorsHandler;
    MotorStatusHandler m_motorStatusHandler;
    ImuDataHandler m_imuDataHandler;
    SetTorsoOrientationHandler m_setTorsoOrientationHandler;
    SetHeadPoseHandler m_setHeadPoseHandler;
    ShutdownHandler m_shutdownHandler;

    RouteCallback m_routeCallback;
    ErrorCallback m_errorCallback;

    PendingMessages m_pendingMessagesByDestination[enum_max<Device>::value + 1];

public:
    explicit SerialCommunicationManager(
        Device device,
        uint32_t acknowledgmentTimeoutMs,
        size_t maximumTrialCount,
        SerialPort& serialPort);

    DECLARE_NOT_COPYABLE(SerialCommunicationManager);
    DECLARE_NOT_MOVABLE(SerialCommunicationManager);

    void setBaseStatusHandler(BaseStatusHandler handler);
    void setButtonPressedHandler(ButtonPressedHandler handler);
    void setSetVolumeHandler(SetVolumeHandler handler);
    void setSetLedColorsHandler(SetLedColorsHandler handler);
    void setMotorStatusHandler(MotorStatusHandler handler);
    void setImuDataHandler(ImuDataHandler handler);
    void setSetTorsoOrientationHandler(SetTorsoOrientationHandler handler);
    void setSetHeadPoseHandler(SetHeadPoseHandler handler);
    void setShutdownHandler(ShutdownHandler handler);

    void setRouteCallback(RouteCallback callback);
    void setErrorCallback(ErrorCallback callback);

    template<class Payload>
    void send(Device destination, const Payload& payload, uint32_t timestampMs);
    template<class Payload>
    void send(Device destination, bool acknowledgmentNeeded, const Payload& payload, uint32_t timestampMs);
    void sendRaw(const uint8_t* data, size_t size);

    void update(uint32_t timestampMs);

private:
    template<class Payload>
    void sendAndAddToPending(const Message<Payload>& message, uint32_t timestampMs);
    template<class Payload>
    void send(const Message<Payload>& message);

    void updatePendingMessages(uint32_t timestampMs);

    bool updateRxPreamble();
    void updateCurrentRxMessageSize();
    void readAndHandleRxMessage(uint8_t messageSize);
    void clearPendingMessage(Device destination, uint16_t messageId);

    void route(Device destination, const uint8_t* data, size_t size);

    void logError(const char* message, tl::optional<MessageType> messageType = tl::nullopt);
};

inline void SerialCommunicationManager::setBaseStatusHandler(BaseStatusHandler handler)
{
    m_baseStatusHandler = handler;
}

inline void SerialCommunicationManager::setButtonPressedHandler(ButtonPressedHandler handler)
{
    m_buttonPressedHandler = handler;
}

inline void SerialCommunicationManager::setSetVolumeHandler(SetVolumeHandler handler)
{
    m_setVolumeHandler = handler;
}

inline void SerialCommunicationManager::setSetLedColorsHandler(SetLedColorsHandler handler)
{
    m_setLedColorsHandler = handler;
}

inline void SerialCommunicationManager::setMotorStatusHandler(MotorStatusHandler handler)
{
    m_motorStatusHandler = handler;
}

inline void SerialCommunicationManager::setImuDataHandler(ImuDataHandler handler)
{
    m_imuDataHandler = handler;
}

inline void SerialCommunicationManager::setSetTorsoOrientationHandler(SetTorsoOrientationHandler handler)
{
    m_setTorsoOrientationHandler = handler;
}

inline void SerialCommunicationManager::setSetHeadPoseHandler(SetHeadPoseHandler handler)
{
    m_setHeadPoseHandler = handler;
}

inline void SerialCommunicationManager::setShutdownHandler(ShutdownHandler handler)
{
    m_shutdownHandler = handler;
}

inline void SerialCommunicationManager::setRouteCallback(RouteCallback callback)
{
    m_routeCallback = callback;
}

inline void SerialCommunicationManager::setErrorCallback(ErrorCallback callback)
{
    m_errorCallback = callback;
}

template<class Payload>
void SerialCommunicationManager::send(Device destination, const Payload& payload, uint32_t timestampMs)
{
    sendAndAddToPending(Message<Payload>(m_device, destination, payload), timestampMs);
}

template<class Payload>
void SerialCommunicationManager::send(
    Device destination,
    bool acknowledgmentNeeded,
    const Payload& payload,
    uint32_t timestampMs)
{
    sendAndAddToPending(Message<Payload>(m_device, destination, acknowledgmentNeeded, payload), timestampMs);
}

template<class Payload>
void SerialCommunicationManager::sendAndAddToPending(const Message<Payload>& message, uint32_t timestampMs)
{
    if (message.header().acknowledgmentNeeded())
    {
        size_t destination = static_cast<size_t>(message.header().destination());
        m_pendingMessagesByDestination[destination].set(message, timestampMs);
    }

    send(message);
}

template<class Payload>
void SerialCommunicationManager::send(const Message<Payload>& message)
{
    static_assert(sizeof(Payload) <= SERIAL_COMMUNICATION_MAXIMUM_PAYLOAD_SIZE, "The payload is too big.");

    uint8_t messageSize = SERIAL_COMMUNICATION_MESSAGE_SIZE_SIZE + MessageHeader::HEADER_SIZE + Payload::PAYLOAD_SIZE +
                          SERIAL_COMMUNICATION_CRC8_SIZE;

    m_txBuffer.clear();
    m_txBuffer.write(messageSize);
    message.header().writeTo(m_txBuffer);
    message.payload().writeTo(m_txBuffer);
    uint8_t crc8Value = crc8(m_txBuffer.dataToRead(), m_txBuffer.sizeToRead());
    m_txBuffer.write(crc8Value);

    m_serialPort.write(SERIAL_COMMUNICATION_PREAMBLE, SERIAL_COMMUNICATION_PREAMBLE_SIZE);
    m_serialPort.write(m_txBuffer.dataToRead(), m_txBuffer.sizeToRead());
}

#endif
