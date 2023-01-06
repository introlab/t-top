#ifndef _DAEMON_WEBSOCKET_SERVER_H_
#define _DAEMON_WEBSOCKET_SERVER_H_

#include <QWebSocketServer>
#include <QWebSocket>
#include "SerialMessages.h"
#include "SerialCommunication.h"
#include "WebSocketProtocolWrapper.h"

class DaemonWebSocketServer : public QWebSocketServer
{
    Q_OBJECT

public:
    DaemonWebSocketServer(QString name, int port, int num_clients = 1, QObject* parent = nullptr);
    void sendBinaryToAll(const QByteArray &data);
    size_t clientCount();
    template<class Payload>
    void send(Device destination, const Payload& payload);

private slots:

    void onNewConnection();
    void processTextMessage(QString message);
    void processBinaryMessage(QByteArray message);
    void socketDisconnected();

private:
    int m_port;
    QList<QWebSocket*> m_clients;
};

template<class Payload>
void DaemonWebSocketServer::send(Device source, const Payload& payload)
{
    // No clients, do nothing...
    if (clientCount() == 0)
    {
        return;
    }

    SerialCommunicationBuffer<SERIAL_COMMUNICATION_BUFFER_SIZE> buffer;
    Message<Payload> message(source, Device::COMPUTER, payload);

    static_assert(sizeof(Payload) <= SERIAL_COMMUNICATION_MAXIMUM_PAYLOAD_SIZE, "The payload is too big.");

    uint8_t messageSize = SERIAL_COMMUNICATION_MESSAGE_SIZE_SIZE + MessageHeader::HEADER_SIZE + Payload::PAYLOAD_SIZE +
                          SERIAL_COMMUNICATION_CRC8_SIZE;
    buffer.clear();
    buffer.write(messageSize);
    message.header().writeTo(buffer);
    message.payload().writeTo(buffer);
    uint8_t crc8Value = crc8(buffer.dataToRead(), buffer.sizeToRead());
    buffer.write(crc8Value);

    // Send to all websockets (in a single message)
    sendBinaryToAll(QByteArray((char*) SERIAL_COMMUNICATION_PREAMBLE, SERIAL_COMMUNICATION_PREAMBLE_SIZE).append(
        QByteArray((char*) buffer.dataToRead(), buffer.sizeToRead())));
}


#endif  // _DAEMON_WEBSOCKET_SERVER_H_
