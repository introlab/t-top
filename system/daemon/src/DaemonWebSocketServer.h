#ifndef _DAEMON_WEBSOCKET_SERVER_H_
#define _DAEMON_WEBSOCKET_SERVER_H_

#include <QWebSocketServer>
#include <QWebSocket>
#include "SerialMessages.h"
#include "WebSocketProtocolWrapper.h"

class DaemonWebSocketServer : public QWebSocketServer
{
    Q_OBJECT

public:
    DaemonWebSocketServer(QString name, int port, int num_clients = 1, QObject* parent = nullptr);
    size_t clientCount();
    template<class Payload>
    void sendToClients(Device destination, const Payload& payload);

signals:
    void newRoute(Device destination, const uint8_t* data, size_t size);

private slots:

    void onNewConnection();
    void socketDisconnected();

private:
    int m_port;
    QList<WebSocketProtocolWrapper*> m_clients;
};

template<class Payload>
void DaemonWebSocketServer::sendToClients(Device source, const Payload& payload)
{
    foreach (auto client, m_clients)
    {
        client->send(source, Device::COMPUTER, payload);
    }
}


#endif  // _DAEMON_WEBSOCKET_SERVER_H_
