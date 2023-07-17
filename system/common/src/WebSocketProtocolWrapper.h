#ifndef _WEBSOCKET_PROTOCOL_WRAPPER_H_
#define _WEBSOCKET_PROTOCOL_WRAPPER_H_

#include <QWebSocket>
#include <QObject>
#include <QTimer>

#include "SerialCommunicationBuffer.h"
#include "SerialMessages.h"
#include "SerialMessagePayloads.h"
#include "SerialCommunicationManager.h"

#ifndef WEBSOCKET_PROTOCOL_WRAPPER_ROS_DEFAULT_CLIENT_PORT
#define WEBSOCKET_PROTOCOL_WRAPPER_ROS_DEFAULT_CLIENT_PORT 48080
#endif

#ifndef WEBSOCKET_PROTOCOL_WRAPPER_CLI_DEFAULT_CLIENT_PORT
#define WEBSOCKET_PROTOCOL_WRAPPER_CLI_DEFAULT_CLIENT_PORT 48081
#endif

#ifndef WEBSOCKET_PROTOCOL_WRAPPER_TRAY_DEFAULT_CLIENT_PORT
#define WEBSOCKET_PROTOCOL_WRAPPER_TRAY_DEFAULT_CLIENT_PORT 48082
#endif

#define STR_IMPL_(x) #x
#define STR(x) STR_IMPL_(x)

class WebSocketProtocolWrapper : public QObject
{
    Q_OBJECT

    static constexpr int CHECK_WEBSOCKET_TIMER_INTERVAL_MS = 1000;

public:
    static constexpr const int ROS_DEFAULT_CLIENT_PORT = WEBSOCKET_PROTOCOL_WRAPPER_ROS_DEFAULT_CLIENT_PORT;
    static constexpr const int CLI_DEFAULT_CLIENT_PORT = WEBSOCKET_PROTOCOL_WRAPPER_CLI_DEFAULT_CLIENT_PORT;
    static constexpr const int TRAY_DEFAULT_CLIENT_PORT = WEBSOCKET_PROTOCOL_WRAPPER_TRAY_DEFAULT_CLIENT_PORT;
    static constexpr const char* ROS_DEFAULT_CLIENT_URL = "ws://localhost:" STR(WEBSOCKET_PROTOCOL_WRAPPER_ROS_DEFAULT_CLIENT_PORT);
    static constexpr const char* CLI_DEFAULT_CLIENT_URL = "ws://localhost:" STR(WEBSOCKET_PROTOCOL_WRAPPER_CLI_DEFAULT_CLIENT_PORT);
    static constexpr const char* TRAY_DEFAULT_CLIENT_URL = "ws://localhost:" STR(WEBSOCKET_PROTOCOL_WRAPPER_TRAY_DEFAULT_CLIENT_PORT);

    explicit WebSocketProtocolWrapper(QWebSocket* websocket, QObject* parent=nullptr);
    explicit WebSocketProtocolWrapper(const QUrl& url, QObject* parent=nullptr);

    template<class Payload>
    void send(Device source, Device destination, const Payload& payload, qint64 timestamp_ms=QDateTime::currentMSecsSinceEpoch());

signals:
    void newBaseStatus(Device source, const BaseStatusPayload& payload);
    void newButtonPressed(Device source, const ButtonPressedPayload& payload);
    void newSetVolume(Device source, const SetVolumePayload& payload);
    void newSetLedColors(Device source, const SetLedColorsPayload& payload);
    void newMotorStatus(Device source, const MotorStatusPayload& payload);
    void newImuData(Device source, const ImuDataPayload& payload);
    void newSetTorsoOrientation(Device source, const SetTorsoOrientationPayload& payload);
    void newSetHeadPose(Device source, const SetHeadPosePayload& payload);
    void newShutdown(Device source, const ShutdownPayload& payload);
    void newRoute(Device destination, const uint8_t* data, size_t size);
    void newError(const char* message, tl::optional<MessageType> messageType);

    void connected();
    void disconnected();

protected slots:
    void binaryMessageReceived(const QByteArray &message);
    void websocketConnected();
    void websocketDisconnected();
    void websocketErrorOccurred(QAbstractSocket::SocketError error);

private:
    void createWebSocketFromUrl(const QUrl& url);

    QWebSocket* m_websocket;
    QTimer* m_websocketCheckTimer;
};

template<class Payload>
void WebSocketProtocolWrapper::send(Device source, Device destination, const Payload& payload, qint64 timestamp_ms)
{
    Q_ASSERT(m_websocket);
    SerialCommunicationBuffer<SERIAL_COMMUNICATION_BUFFER_SIZE> buffer;
    Message<Payload> message(source, destination, payload);

    static_assert(sizeof(Payload) <= SERIAL_COMMUNICATION_MAXIMUM_PAYLOAD_SIZE, "The payload is too big.");

    buffer.clear();
    message.header().writeTo(buffer);
    message.payload().writeTo(buffer);

    //Send to websocket (in a single message, with no preamble and no CRC8)
    m_websocket->sendBinaryMessage(QByteArray((char*) buffer.dataToRead(), buffer.sizeToRead()));
    m_websocket->flush();
}


#endif // _WEBSOCKET_PROTOCOL_WRAPPER_H_
