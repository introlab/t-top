#ifndef _WEBSOCKET_SERIAL_WRAPPER_H_
#define _WEBSOCKET_SERIAL_WRAPPER_H_

#include "SerialCommunication.h"
#include <QObject>
#include <WebSocketDaemonClient.h>

class WebSocketSerialWrapper : public QObject, public SerialPort
{
    Q_OBJECT

public:
    WebSocketSerialWrapper(QObject* parent = nullptr);
    void read(SerialCommunicationBufferView& buffer) override;
    void write(const uint8_t* data, size_t size) override;
    void connectTo(const QUrl& url);

signals:
    void webSocketConnected();
    void webSocketDisconnected();
    void readyRead();

private:
    WebSocketDaemonClient m_webSocketClient;
};

#endif // _WEBSOCKET_SERIAL_WRAPPER_H_
