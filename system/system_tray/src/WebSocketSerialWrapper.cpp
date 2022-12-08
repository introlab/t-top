#include "WebSocketSerialWrapper.h"
#include <algorithm>

WebSocketSerialWrapper::WebSocketSerialWrapper(QObject *parent)
    : QObject(parent), m_webSocketClient(this)
{
    // Signal on signal
    connect(&m_webSocketClient, &WebSocketDaemonClient::webSocketConnected, this, &WebSocketSerialWrapper::webSocketConnected);
    connect(&m_webSocketClient, &WebSocketDaemonClient::webSocketDisconnected, this, &WebSocketSerialWrapper::webSocketDisconnected);
    connect(&m_webSocketClient, &WebSocketDaemonClient::readyRead, this, &WebSocketSerialWrapper::readyRead);
}

void WebSocketSerialWrapper::read(SerialCommunicationBufferView &buffer)
{
    size_t read_size = std::min(buffer.sizeToWrite(), (size_t)m_webSocketClient.binaryBytesAvailable());
    qDebug() << "WebSocketSerialWrapper::read size = " << read_size;

    QByteArray data = m_webSocketClient.readBinaryBytes(read_size);

    if (data.size() == read_size)
    {
        buffer.write(reinterpret_cast<const uint8_t*>(data.constData()), data.size());
    }
    else
    {
        qDebug() << "Reading buffer... error expected: " << read_size << " got: " << data.size();
    }
}

void WebSocketSerialWrapper::write(const uint8_t *data, size_t size)
{
    qDebug() <<  "WebSocketSerialWrapper::write size = " << size;
    m_webSocketClient.write(data, size);

}

void WebSocketSerialWrapper::connectTo(const QUrl &url)
{
    m_webSocketClient.connectTo(url);
}
