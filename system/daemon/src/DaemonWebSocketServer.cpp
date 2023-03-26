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
    QWebSocket* websocket = nextPendingConnection();
    if (websocket)
    {
        qDebug() << serverName() << "onNewConnection: " << websocket;
        WebSocketProtocolWrapper* wrapper = new WebSocketProtocolWrapper(websocket, this);
        connect(wrapper, &WebSocketProtocolWrapper::disconnected, this, &DaemonWebSocketServer::socketDisconnected);
        connect(wrapper, &WebSocketProtocolWrapper::newRoute, this, &DaemonWebSocketServer::newRoute);
        m_clients.append(wrapper);
    }
}


void DaemonWebSocketServer::socketDisconnected()
{
    WebSocketProtocolWrapper* wrapper = qobject_cast<WebSocketProtocolWrapper*>(sender());
    qDebug() << serverName() << "socketDisconnected: " << wrapper;
    m_clients.removeAll(wrapper);
    wrapper->deleteLater();
}

size_t DaemonWebSocketServer::clientCount()
{
    return m_clients.size();
}
