#include "WebSocketDaemonClient.h"
#include <QByteArray>
#include <algorithm>

WebSocketDaemonClient::WebSocketDaemonClient(QObject* parent) : QObject{parent}, m_webSocket(nullptr)
{
    m_webSocket = new QWebSocket(QString(), QWebSocketProtocol::VersionLatest, parent);
    connectTo("localhost", 8082);
}

void WebSocketDaemonClient::connectTo(const QUrl& url)
{
    Q_ASSERT(m_webSocket);
    // Signal to slot
    connect(m_webSocket, &QWebSocket::connected, this, &WebSocketDaemonClient::onSocketConnected);
    connect(m_webSocket, &QWebSocket::disconnected, this, &WebSocketDaemonClient::onSocketDisconnected);
    connect(m_webSocket, &QWebSocket::binaryMessageReceived, this, &WebSocketDaemonClient::onInternalBinaryMessageReceived);

    // Signal to signal
    connect(m_webSocket, &QWebSocket::binaryMessageReceived, this, &WebSocketDaemonClient::binaryMessageReceived);
    connect(m_webSocket, &QWebSocket::textMessageReceived, this, &WebSocketDaemonClient::textMessageReceived);
    connect(m_webSocket, &QWebSocket::connected, this, &WebSocketDaemonClient::webSocketConnected);
    connect(m_webSocket, &QWebSocket::disconnected, this, &WebSocketDaemonClient::webSocketDisconnected);
    m_webSocket->open(url);

}

void WebSocketDaemonClient::connectTo(const QString& hostname, int port)
{
    connectTo(QUrl(QString("ws://%1:%2").arg(hostname).arg(port)));
}

size_t WebSocketDaemonClient::sendBinaryMessage(const QByteArray &data)
{
    Q_ASSERT(m_webSocket);
    return m_webSocket->sendBinaryMessage(data);
}

void WebSocketDaemonClient::onSocketConnected()
{
    qDebug() << "WebSocketDaemonClient::socketConnected()";
}

void WebSocketDaemonClient::onSocketDisconnected()
{
    qDebug() << "WebSocketDaemonClient::socketDisconnected()";
}

void WebSocketDaemonClient::onInternalBinaryMessageReceived(const QByteArray &message)
{
    m_internalBuffer.append(message);
    emit readyRead();
}


size_t WebSocketDaemonClient::binaryBytesAvailable()
{
    Q_ASSERT(m_webSocket);
    return m_internalBuffer.size();
}

void WebSocketDaemonClient::write(const uint8_t *data, size_t size)
{
    Q_ASSERT(m_webSocket);
    m_webSocket->sendBinaryMessage(QByteArray(reinterpret_cast<const char*>(data), size));
}

QByteArray WebSocketDaemonClient::readBinaryBytes(size_t max_size)
{
    auto allowed_size = std::min(max_size, (size_t) m_internalBuffer.size());
    QByteArray result = m_internalBuffer.left(allowed_size);
    m_internalBuffer.remove(0, allowed_size);
    return result;
}
