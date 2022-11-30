#ifndef _SYSTEM_TRAY_APP_H_
#define _SYSTEM_TRAY_APP_H_

#include <QApplication>
#include <QObject>
#include "SystemTrayIcon.h"
#include "WebSocketDaemonClient.h"

class SystemTrayApp : public QApplication
{
    Q_OBJECT
public:
    SystemTrayApp(int argc, char* argv[]);

private:
    SystemTrayIcon* m_trayIcon;
    WebSocketDaemonClient* m_webSocketClient;
};

#endif  // _SYSTEM_TRAY_APP_H_
