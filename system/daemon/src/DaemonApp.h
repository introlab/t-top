#ifndef _DAEMON_APP_H_
#define _DAEMON_APP_H_

#include <QCoreApplication>
#include <QWebSocketServer>
#include <QWebSocket>
#include <QTimer>
#include <QList>
#include "DaemonWebSocketServer.h"
#include "DaemonSerialManager.h"


class DeamonApp : public QCoreApplication {
    Q_OBJECT

public:
    DeamonApp(int argc, char* argv[]);

private slots:


private:

    void setupWebSocketServers();
    void setupSerialManager();

    QList<DaemonWebSocketServer*> m_webSocketServers;
    DaemonSerialManager* m_serialManager;

};

#endif //_DAEMON_APP_H_
