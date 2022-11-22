#ifndef _DAEMON_WEBSOCKET_SERVER_H_
#define _DAEMON_WEBSOCKET_SERVER_H_

#include <QWebSocketServer>
#include <QWebSocket>


class DaemonWebSocketServer : public QWebSocketServer
{
    Q_OBJECT

public:
    DaemonWebSocketServer(QString name, int port, int num_clients = 1, QObject* parent = nullptr);

private slots:

    void onNewConnection();
    void processTextMessage(QString message);
    void processBinaryMessage(QByteArray message);
    void socketDisconnected();


private:
    int m_port;
    QList<QWebSocket*> m_clients;
};


#endif  // _DAEMON_WEBSOCKET_SERVER_H_
