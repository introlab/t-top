#include "WebSocketProtocolWrapper.h"


WebSocketProtocolWrapper::WebSocketProtocolWrapper(QWebSocket* websocket, QObject* parent)
    : QObject(parent),
      m_websocket(websocket),
      m_websocketCheckTimer(nullptr)
{
    Q_ASSERT(m_websocket);
    // Connect signals
    connect(m_websocket, &QWebSocket::binaryMessageReceived, this, &WebSocketProtocolWrapper::binaryMessageReceived);
    connect(m_websocket, &QWebSocket::connected, this, &WebSocketProtocolWrapper::websocketConnected);
    connect(m_websocket, &QWebSocket::disconnected, this, &WebSocketProtocolWrapper::disconnected);
}

WebSocketProtocolWrapper::WebSocketProtocolWrapper(const QUrl& url, QObject* parent)
    : QObject(parent),
      m_websocket(nullptr)
{
    createWebSocketFromUrl(url);

    m_websocketCheckTimer = new QTimer(this);
    connect(
        m_websocketCheckTimer,
        &QTimer::timeout,
        this,
        [url, this]()
        {
            if (m_websocket == nullptr)
            {
                createWebSocketFromUrl(url);
            }
        });
    m_websocketCheckTimer->start(CHECK_WEBSOCKET_TIMER_INTERVAL_MS);
}

void WebSocketProtocolWrapper::binaryMessageReceived(const QByteArray& message)
{
    // We should have a full message without the preamble
    const uint8_t* dataToRoute = reinterpret_cast<const uint8_t*>(message.constData());
    size_t messageSize = message.size();

    SerialCommunicationBuffer<SERIAL_COMMUNICATION_BUFFER_SIZE> m_rxBuffer;
    m_rxBuffer.write(dataToRoute, messageSize);

    auto header = MessageHeader::readFrom(m_rxBuffer);
    if (header == tl::nullopt)
    {
        return;
    }

    if (header->destination() != Device::COMPUTER)
    {
        emit newRoute(header->destination(), dataToRoute, messageSize);
        return;
    }

    switch (header->messageType())
    {
        case MessageType::ACKNOWLEDGMENT:
        {
            AcknowledgmentPayload::readFrom(m_rxBuffer);
            emit newError("Acknowledgment payload not supported by websocket", header->messageType());
            break;
        }

        case MessageType::BASE_STATUS:
        {
            auto payload = BaseStatusPayload::readFrom(m_rxBuffer);
            if (payload.has_value())
            {
                emit newBaseStatus(header->source(), *payload);
            }
            break;
        }

        case MessageType::BUTTON_PRESSED:
        {
            auto payload = ButtonPressedPayload::readFrom(m_rxBuffer);
            if (payload.has_value())
            {
                emit newButtonPressed(header->source(), *payload);
            }
            break;
        }

        case MessageType::SET_VOLUME:
        {
            auto payload = SetVolumePayload::readFrom(m_rxBuffer);
            if (payload.has_value())
            {
                emit newSetVolume(header->source(), *payload);
            }
            break;
        }

        case MessageType::SET_LED_COLORS:
        {
            auto payload = SetLedColorsPayload::readFrom(m_rxBuffer);
            if (payload.has_value())
            {
                emit newSetLedColors(header->source(), *payload);
            }
            break;
        }

        case MessageType::MOTOR_STATUS:
        {
            auto payload = MotorStatusPayload::readFrom(m_rxBuffer);
            if (payload.has_value())
            {
                emit newMotorStatus(header->source(), *payload);
            }
            break;
        }

        case MessageType::IMU_DATA:
        {
            auto payload = ImuDataPayload::readFrom(m_rxBuffer);
            if (payload.has_value())
            {
                emit newImuData(header->source(), *payload);
            }
            break;
        }

        case MessageType::SET_TORSO_ORIENTATION:
        {
            auto payload = SetTorsoOrientationPayload::readFrom(m_rxBuffer);
            if (payload.has_value())
            {
                emit newSetTorsoOrientation(header->source(), *payload);
            }
            break;
        }

        case MessageType::SET_HEAD_POSE:
        {
            auto payload = SetHeadPosePayload::readFrom(m_rxBuffer);
            if (payload.has_value())
            {
                emit newSetHeadPose(header->source(), *payload);
            }
            break;
        }

        case MessageType::SHUTDOWN:
        {
            auto payload = ShutdownPayload::readFrom(m_rxBuffer);
            if (payload.has_value())
            {
                emit newShutdown(header->source(), *payload);
            }
            break;
        }
    }
}

void WebSocketProtocolWrapper::websocketConnected()
{
    qDebug() << "WebSocketProtocolWrapper::websocketConnected() " << m_websocket;
    emit connected();
}

void WebSocketProtocolWrapper::websocketDisconnected()
{
    qDebug() << "WebSocketProtocolWrapper::websocketDisconnected() " << m_websocket;
    m_websocket->deleteLater();
    m_websocket = nullptr;
    emit disconnected();
}

void WebSocketProtocolWrapper::websocketErrorOccurred(QAbstractSocket::SocketError error)
{
    websocketDisconnected();
}

void WebSocketProtocolWrapper::createWebSocketFromUrl(const QUrl& url)
{
    if (m_websocket != nullptr)
    {
        m_websocket->deleteLater();
    }
    m_websocket = new QWebSocket(QString(), QWebSocketProtocol::VersionLatest, this);

    connect(m_websocket, &QWebSocket::binaryMessageReceived, this, &WebSocketProtocolWrapper::binaryMessageReceived);
    connect(m_websocket, &QWebSocket::connected, this, &WebSocketProtocolWrapper::websocketConnected);
    connect(m_websocket, &QWebSocket::disconnected, this, &WebSocketProtocolWrapper::websocketDisconnected);
    connect(
        m_websocket,
        QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::error),
        this,
        &WebSocketProtocolWrapper::websocketErrorOccurred);

    m_websocket->open(url);
}
