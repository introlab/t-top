#include "WebSocketDaemonClient.h"

WebSocketDaemonClient::WebSocketDaemonClient(QObject* parent) : QObject{parent}, m_webSocket(nullptr)
{
    m_webSocket = new QWebSocket(QString(), QWebSocketProtocol::VersionLatest, parent);
    connectTo("localhost", 8082);
}

void WebSocketDaemonClient::connectTo(const QUrl& url)
{
    if (m_webSocket)
    {
        connect(m_webSocket, &QWebSocket::connected, this, &WebSocketDaemonClient::socketConnected);
        connect(m_webSocket, &QWebSocket::disconnected, this, &WebSocketDaemonClient::socketDisconnected);
        m_webSocket->open(url);
    }
}

void WebSocketDaemonClient::connectTo(const QString& hostname, int port)
{
    connectTo(QUrl(QString("ws://%1:%2").arg(hostname).arg(port)));
}

void WebSocketDaemonClient::socketConnected()
{
    qDebug() << "WebSocketDaemonClient::socketConnected()";
}

void WebSocketDaemonClient::socketDisconnected()
{
    qDebug() << "WebSocketDaemonClient::socketDisconnected()";
}
