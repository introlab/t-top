#ifndef _WEBSOCKET_PROTOCOL_WRAPPER_H_
#define _WEBSOCKET_PROTOCOL_WRAPPER_H_

#include <QWebSocket>
#include <QObject>
#include "SerialCommunicationBuffer.h"
#include "SerialMessages.h"
#include "SerialMessagePayloads.h"
#include "SerialCommunicationManager.h"



class WebSocketProtocolWrapper : public QObject
{
    Q_OBJECT
public:
    WebSocketProtocolWrapper(QWebSocket* websocket);

    template<class Payload>
    void send(Device destination, const Payload& payload);

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

protected slots:
    void binaryMessageReceived(const QByteArray &message);

private:
    QWebSocket *m_websocket;
};

template<class Payload>
void WebSocketProtocolWrapper::send(Device source, const Payload& payload)
{
    Q_ASSERT(m_websocket);
    SerialCommunicationBuffer<SERIAL_COMMUNICATION_BUFFER_SIZE> buffer;
    Message<Payload> message(source, Device::COMPUTER, payload);

    static_assert(sizeof(Payload) <= SERIAL_COMMUNICATION_MAXIMUM_PAYLOAD_SIZE, "The payload is too big.");

    buffer.clear();
    message.header().writeTo(buffer);
    message.payload().writeTo(buffer);

    //Send to websocket (in a single message, with no preamble and no CRC8)
    m_websocket->sendBinaryMessage(QByteArray((char*) buffer.dataToRead(), buffer.sizeToRead()));
}


#endif // _WEBSOCKET_PROTOCOL_WRAPPER_H_
