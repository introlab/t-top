#include "WebSocketProtocolWrapper.h"



WebSocketProtocolWrapper::WebSocketProtocolWrapper(QWebSocket *websocket, QObject *parent)
    : QObject(parent), m_websocket(websocket)
{
    Q_ASSERT(m_websocket);
    // Connect signals
    connect(m_websocket, &QWebSocket::binaryMessageReceived, this, &WebSocketProtocolWrapper::binaryMessageReceived);
    connect(m_websocket, &QWebSocket::connected, this, &WebSocketProtocolWrapper::websocketConnected);
    connect(m_websocket, &QWebSocket::disconnected, this, &WebSocketProtocolWrapper::disconnected);
}

WebSocketProtocolWrapper::WebSocketProtocolWrapper(const QUrl url, QObject *parent)
{
    m_websocket = new QWebSocket(QString(), QWebSocketProtocol::VersionLatest, parent);
    // Connect signals
    connect(m_websocket, &QWebSocket::binaryMessageReceived, this, &WebSocketProtocolWrapper::binaryMessageReceived);
    connect(m_websocket, &QWebSocket::connected, this, &WebSocketProtocolWrapper::websocketConnected);
    connect(m_websocket, &QWebSocket::disconnected, this, &WebSocketProtocolWrapper::disconnected);

    qDebug() <<"WebSocketProtocolWrapper::WebSocketProtocolWrapper connecting to :" << url;
    m_websocket->open(url);
}

QWebSocket *WebSocketProtocolWrapper::getWebSocket()
{
    return m_websocket;
}

void WebSocketProtocolWrapper::binaryMessageReceived(const QByteArray &message)
{
    // We should have a full message without the preamble
    const uint8_t* dataToRoute = reinterpret_cast<const uint8_t*>(message.constData());
    size_t messageSize = message.size();

    SerialCommunicationBuffer<SERIAL_COMMUNICATION_BUFFER_SIZE> m_rxBuffer;
    m_rxBuffer.write(dataToRoute, messageSize);

    // TODO check optional ?
    auto header = *MessageHeader::readFrom(m_rxBuffer);

    // Not for computer? route message...
    if (header.destination() !=  Device::COMPUTER)
    {
        emit newRoute(header.destination(), dataToRoute, messageSize);
        return;
    }

    // TODO check optional ?
    switch (header.messageType())
    {
        case MessageType::ACKNOWLEDGMENT:
        {
            auto payload = *AcknowledgmentPayload::readFrom(m_rxBuffer);
            emit newError("Acknowledgment payload not supported by websocket", header.messageType());
            break;
        }

        case MessageType::BASE_STATUS:
        {
            auto payload = *BaseStatusPayload::readFrom(m_rxBuffer);
            emit newBaseStatus(header.source(), payload);
            break;
        }

        case MessageType::BUTTON_PRESSED:
        {
            auto payload = *ButtonPressedPayload::readFrom(m_rxBuffer);
            emit newButtonPressed(header.source(), payload);
            break;
        }

        case MessageType::SET_VOLUME:
        {
            auto payload = *SetVolumePayload::readFrom(m_rxBuffer);
            emit newSetVolume(header.source(), payload);
            break;
        }

        case MessageType::SET_LED_COLORS:
        {
            auto payload = *SetLedColorsPayload::readFrom(m_rxBuffer);
            emit newSetLedColors(header.source(), payload);
            break;
        }

        case MessageType::MOTOR_STATUS:
        {
            auto payload = *MotorStatusPayload::readFrom(m_rxBuffer);
            emit newMotorStatus(header.source(), payload);
            break;
        }

        case MessageType::IMU_DATA:
        {
            auto payload =  *ImuDataPayload::readFrom(m_rxBuffer);
            emit newImuData(header.source(), payload);
            break;
        }

        case MessageType::SET_TORSO_ORIENTATION:
        {
            auto payload = *SetTorsoOrientationPayload::readFrom(m_rxBuffer);
            emit newSetTorsoOrientation(header.source(), payload);
            break;
        }

        case MessageType::SET_HEAD_POSE:
        {
            auto payload = *SetHeadPosePayload::readFrom(m_rxBuffer);
            emit newSetHeadPose(header.source(), payload);
            break;
        }

        case MessageType::SHUTDOWN:
        {
            auto payload = *ShutdownPayload::readFrom(m_rxBuffer);
            emit newShutdown(header.source(), payload);
            break;
        }
    }
}

void WebSocketProtocolWrapper::websocketConnected()
{
    qDebug() << "WebSocketProtocolWrapper::websocketConnected() " << m_websocket;
}
