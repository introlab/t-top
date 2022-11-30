#include "DaemonWebSocketServer.h"


DaemonWebSocketServer::DaemonWebSocketServer(QString name, int port, int num_clients, QObject* parent)
    : QWebSocketServer(name, QWebSocketServer::NonSecureMode, parent),
      m_port(port)
{
    // Allowing only num_client connections
    setMaxPendingConnections(num_clients);

    // Listen only on localhost
    if (listen(QHostAddress::LocalHost, m_port))
    {
        connect(this, &QWebSocketServer::newConnection, this, &DaemonWebSocketServer::onNewConnection);
        qDebug() << serverName() << "listening on port " << m_port;
    }
}

void DaemonWebSocketServer::onNewConnection()
{
    // Accepting only on localhost.
    QWebSocket* socket = nextPendingConnection();
    if (socket)
    {
        connect(socket, &QWebSocket::textMessageReceived, this, &DaemonWebSocketServer::processTextMessage);
        connect(socket, &QWebSocket::binaryMessageReceived, this, &DaemonWebSocketServer::processBinaryMessage);
        connect(socket, &QWebSocket::disconnected, this, &DaemonWebSocketServer::socketDisconnected);
        qDebug() << serverName() << "onNewConnection: " << socket;
        m_clients.append(socket);
    }
}

void DaemonWebSocketServer::processTextMessage(QString message)
{
    QWebSocket* client = qobject_cast<QWebSocket*>(sender());
    qDebug() << serverName() << "processTextMessage: " << client;
}

void DaemonWebSocketServer::processBinaryMessage(QByteArray message)
{
    QWebSocket* client = qobject_cast<QWebSocket*>(sender());
    qDebug() << serverName() << "processBinaryMessage: " << client;
}

void DaemonWebSocketServer::socketDisconnected()
{
    QWebSocket* client = qobject_cast<QWebSocket*>(sender());
    qDebug() << serverName() << "socketDisconnected: " << client;
}
