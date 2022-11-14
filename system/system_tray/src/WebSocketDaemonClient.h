#ifndef _WEBSOCKET_DAEMON_CLIENT_H_
#define _WEBSOCKET_DAEMON_CLIENT_H_

#include <QObject>
#include <QWebSocket>
#include <QUrl>

class WebSocketDaemonClient : public QObject
{
    Q_OBJECT
public:
    explicit WebSocketDaemonClient(QObject *parent = nullptr);

public slots:
    void connectTo(const QUrl &url);

    void connectTo(const QString &hostname, int port);

signals:

private slots:

    void socketConnected();
    void socketDisconnected();

private:
    QWebSocket *m_webSocket;

};

#endif // _WEBSOCKET_DAEMON_CLIENT_H_
