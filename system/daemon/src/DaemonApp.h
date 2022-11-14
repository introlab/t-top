#ifndef _DAEMON_APP_H_
#define _DAEMON_APP_H_

#include <QCoreApplication>
#include <QWebSocketServer>
#include <QWebSocket>
#include <QTimer>


class DeamonApp : public QCoreApplication {
    Q_OBJECT

public:
    DeamonApp(int argc, char* argv[]);

private slots:

    void onNewConnection()
    {

        //Accept only localhost.
        QWebSocket *socket = m_websocketServer->nextPendingConnection();
        connect(socket, &QWebSocket::textMessageReceived, this, &DeamonApp::processTextMessage);
        connect(socket, &QWebSocket::binaryMessageReceived, this, &DeamonApp::processBinaryMessage);
        connect(socket, &QWebSocket::disconnected, this, &DeamonApp::socketDisconnected);

        //m_clients << pSocket;
    }

    void processTextMessage(QString message)
    {
        QWebSocket *client = qobject_cast<QWebSocket *>(sender());
        qDebug() << "processTextMessage:" << client;
    }

    void processBinaryMessage(QByteArray message)
    {
        QWebSocket *client = qobject_cast<QWebSocket *>(sender());
        qDebug() << "processBinaryMessage:" << client;
    }

    void socketDisconnected()
    {
        QWebSocket *client = qobject_cast<QWebSocket *>(sender());
        qDebug() << "socketDisconnected:" << client;
    }

private:

    void create_websocket_server(int port)
    {
        m_websocketServer = new QWebSocketServer("TTOPDaemon",QWebSocketServer::NonSecureMode,this);

        if (m_websocketServer->listen(QHostAddress::Any, 8080)) {

            qDebug() << "DeamonApp listening on port" << port;
            connect(m_websocketServer, &QWebSocketServer::newConnection,
                    this, &DeamonApp::onNewConnection);
            //connect(m_websocketServer, &QWebSocketServer::closed, this, &DeamonApp::closed);
        }


    }

    QWebSocketServer *m_websocketServer;
};

#endif //_DAEMON_APP_H_
