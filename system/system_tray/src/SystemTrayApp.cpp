#include "SystemTrayApp.h"

SystemTrayApp::SystemTrayApp(int argc, char* argv[]) : QApplication(argc, argv)
{
    m_trayIcon = new SystemTrayIcon(this);
    m_trayIcon->show();

    m_webSocketClient = new WebSocketDaemonClient(this);
}
