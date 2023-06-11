#include "SerialCommunicationManager.h"

using namespace std;

PendingMessages::PendingMessages() {}

void PendingMessages::clear(uint16_t messageId)
{
    forEach(
        [messageId](auto& message)
        {
            if (message.has_value() && message->message.header().messageId() == messageId)
            {
                message = tl::nullopt;
            }
        });
}

SerialCommunicationManager::SerialCommunicationManager(
    Device device,
    uint32_t acknowledgmentTimeoutMs,
    size_t maximumTrialCount,
    SerialPort& serialPort)
    : m_device(device),
      m_acknowledgmentTimeoutMs(acknowledgmentTimeoutMs),
      m_maximumTrialCount(maximumTrialCount),
      m_serialPort(serialPort),
      m_isWaitingRxPreamble(true),
      m_baseStatusHandler(nullptr),
      m_buttonPressedHandler(nullptr),
      m_setVolumeHandler(nullptr),
      m_setLedColorsHandler(nullptr),
      m_motorStatusHandler(nullptr),
      m_imuDataHandler(nullptr),
      m_setTorsoOrientationHandler(nullptr),
      m_setHeadPoseHandler(nullptr),
      m_shutdownHandler(nullptr),
      m_routeCallback(nullptr),
      m_errorCallback(nullptr)
{
}

void SerialCommunicationManager::sendRaw(const uint8_t* data, size_t size)
{
    m_serialPort.write(SERIAL_COMMUNICATION_PREAMBLE, SERIAL_COMMUNICATION_PREAMBLE_SIZE);
    m_serialPort.write(data, size);
}

void SerialCommunicationManager::update(uint32_t timestampMs)
{
    SerialCommunicationBufferView m_rxBufferView(m_rxBuffer);
    m_serialPort.read(m_rxBufferView);

    size_t lastSizeToRead;
    do
    {
        lastSizeToRead = m_rxBuffer.sizeToRead();

        if (m_isWaitingRxPreamble)
        {
            while (!updateRxPreamble()) {}
        }

        if (!m_isWaitingRxPreamble && m_currentRxMessageSize == tl::nullopt)
        {
            updateCurrentRxMessageSize();
        }
        if (!m_isWaitingRxPreamble && m_currentRxMessageSize.has_value() &&
            m_rxBuffer.sizeToRead() >= *m_currentRxMessageSize)
        {
            readAndHandleRxMessage(*m_currentRxMessageSize);
            m_rxBuffer.moveToBeginning();
            m_isWaitingRxPreamble = true;
            m_currentRxMessageSize = tl::nullopt;
        }

    } while (m_rxBuffer.sizeToRead() != lastSizeToRead);

    updatePendingMessages(timestampMs);
}

void SerialCommunicationManager::updatePendingMessages(uint32_t timestampMs)
{
    for (size_t destination = 0; destination <= enum_max<Device>::value; destination++)
    {
        PendingMessages& pendingMessages = m_pendingMessagesByDestination[destination];
        pendingMessages.forEach(
            [this, timestampMs](auto& message)
            {
                if (message.has_value() && (timestampMs - message->timestampMs) >= m_acknowledgmentTimeoutMs)
                {
                    message->trialCount++;
                    message->timestampMs = timestampMs;
                    if (message->trialCount > m_maximumTrialCount)
                    {
                        message = tl::nullopt;
                        this->logError(
                            "Too many trials: The message is dropped.",
                            message->message.header().messageType());
                    }
                    else
                    {
                        this->send(message->message);
                    }
                }
            });
    }
}

bool SerialCommunicationManager::updateRxPreamble()
{
    if (m_rxBuffer.sizeToRead() >= SERIAL_COMMUNICATION_PREAMBLE_SIZE)
    {
        for (size_t i = 0; i < SERIAL_COMMUNICATION_PREAMBLE_SIZE; i++)
        {
            auto value = m_rxBuffer.read<uint8_t>();
            if (value != SERIAL_COMMUNICATION_PREAMBLE_BYTE)
            {
                m_rxBuffer.moveToBeginning();
                return false;
            }
        }

        m_isWaitingRxPreamble = false;
    }
    return true;
}

void SerialCommunicationManager::updateCurrentRxMessageSize()
{
    if (m_rxBuffer.sizeToRead() >= 1)
    {
        m_currentRxMessageSize = m_rxBuffer.dataToRead()[0];
    }
}

#undef CHECK_BUFFER_READ
#define CHECK_BUFFER_READ(variable, code)                                                                              \
    auto variable##Optional = (code);                                                                                  \
    if (variable##Optional == tl::nullopt)                                                                             \
    {                                                                                                                  \
        logError("Read Failure");                                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    auto variable = *variable##Optional;

#define CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(device, handler, header, payload)                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        auto expectedCrc8Value = m_rxBuffer.read<uint8_t>();                                                           \
        if (expectedCrc8Value != crc8Value)                                                                            \
        {                                                                                                              \
            logError("CRC8 Error", header.messageType());                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        if (header.acknowledgmentNeeded())                                                                             \
        {                                                                                                              \
            send(Message<AcknowledgmentPayload>(device, header.source(), AcknowledgmentPayload{header.messageId()}));  \
        }                                                                                                              \
        if ((handler) != nullptr)                                                                                      \
        {                                                                                                              \
            handler(header.source(), payload);                                                                         \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            logError("Not handled message", header.messageType());                                                     \
        }                                                                                                              \
    } while (false)

void SerialCommunicationManager::readAndHandleRxMessage(uint8_t messageSize)
{
    const uint8_t* dataToRoute = m_rxBuffer.dataToRead();

    uint8_t crc8Value = crc8(m_rxBuffer.dataToRead(), messageSize - SERIAL_COMMUNICATION_CRC8_SIZE);
    m_rxBuffer.read<uint8_t>();  // Read the message size
    CHECK_BUFFER_READ(header, MessageHeader::readFrom(m_rxBuffer));

    if (header.destination() != m_device)
    {
        route(header.destination(), dataToRoute, messageSize);
        return;
    }

    switch (header.messageType())
    {
        case MessageType::ACKNOWLEDGMENT:
        {
            CHECK_BUFFER_READ(payload, AcknowledgmentPayload::readFrom(m_rxBuffer));
            auto expectedCrc8Value = m_rxBuffer.read<uint8_t>();
            if (expectedCrc8Value == crc8Value)
            {
                clearPendingMessage(header.source(), payload.receivedMessageId);
            }
            else
            {
                logError("CRC8 error", header.messageType());
            }

            return;
        }

        case MessageType::BASE_STATUS:
        {
            CHECK_BUFFER_READ(payload, BaseStatusPayload::readFrom(m_rxBuffer));
            CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(m_device, m_baseStatusHandler, header, payload);
            break;
        }

        case MessageType::BUTTON_PRESSED:
        {
            CHECK_BUFFER_READ(payload, ButtonPressedPayload::readFrom(m_rxBuffer));
            CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(m_device, m_buttonPressedHandler, header, payload);
            break;
        }

        case MessageType::SET_VOLUME:
        {
            CHECK_BUFFER_READ(payload, SetVolumePayload::readFrom(m_rxBuffer));
            CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(m_device, m_setVolumeHandler, header, payload);
            break;
        }

        case MessageType::SET_LED_COLORS:
        {
            CHECK_BUFFER_READ(payload, SetLedColorsPayload::readFrom(m_rxBuffer));
            CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(m_device, m_setLedColorsHandler, header, payload);
            break;
        }

        case MessageType::MOTOR_STATUS:
        {
            CHECK_BUFFER_READ(payload, MotorStatusPayload::readFrom(m_rxBuffer));
            CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(m_device, m_motorStatusHandler, header, payload);
            break;
        }

        case MessageType::IMU_DATA:
        {
            CHECK_BUFFER_READ(payload, ImuDataPayload::readFrom(m_rxBuffer));
            CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(m_device, m_imuDataHandler, header, payload);
            break;
        }

        case MessageType::SET_TORSO_ORIENTATION:
        {
            CHECK_BUFFER_READ(payload, SetTorsoOrientationPayload::readFrom(m_rxBuffer));
            CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(m_device, m_setTorsoOrientationHandler, header, payload);
            break;
        }

        case MessageType::SET_HEAD_POSE:
        {
            CHECK_BUFFER_READ(payload, SetHeadPosePayload::readFrom(m_rxBuffer));
            CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(m_device, m_setHeadPoseHandler, header, payload);
            break;
        }

        case MessageType::SHUTDOWN:
        {
            CHECK_BUFFER_READ(payload, ShutdownPayload::readFrom(m_rxBuffer));
            CHECK_CRC8_SEND_ACK_AND_CALL_HANDLER(m_device, m_shutdownHandler, header, payload);
            break;
        }
    }
}

void SerialCommunicationManager::clearPendingMessage(Device destination, uint16_t messageId)
{
    m_pendingMessagesByDestination[static_cast<size_t>(destination)].clear(messageId);
}

void SerialCommunicationManager::route(Device destination, const uint8_t* data, size_t size)
{
    if (m_routeCallback != nullptr)
    {
        m_routeCallback(destination, data, size);
    }
    else
    {
        logError("No Route Callback: The message is dropped.");
    }
}

void SerialCommunicationManager::logError(const char* message, tl::optional<MessageType> messageType)
{
    if (m_errorCallback != nullptr)
    {
        m_errorCallback(message, messageType);
    }
}
