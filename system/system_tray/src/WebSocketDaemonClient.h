#ifndef _WEBSOCKET_DAEMON_CLIENT_H_
#define _WEBSOCKET_DAEMON_CLIENT_H_

#include <QObject>
#include <QWebSocket>
#include <QUrl>

class WebSocketDaemonClient : public QObject
{
    Q_OBJECT
public:
    explicit WebSocketDaemonClient(QObject* parent = nullptr);

    size_t binaryBytesAvailable();
    void write(const uint8_t *data, size_t size);
    QByteArray readBinaryBytes(size_t max_size);

public slots:
    void connectTo(const QUrl& url);
    void connectTo(const QString& hostname, int port);
    size_t sendBinaryMessage(const QByteArray &data);

signals:
    void binaryMessageReceived(const QByteArray &message);
    void textMessageReceived(const QString &message);
    void webSocketConnected();
    void webSocketDisconnected();
    void readyRead();

private slots:

    void onSocketConnected();
    void onSocketDisconnected();
    void onInternalBinaryMessageReceived(const QByteArray &message);

private:
    QWebSocket* m_webSocket;
    QByteArray m_internalBuffer;
};

#endif  // _WEBSOCKET_DAEMON_CLIENT_H_
