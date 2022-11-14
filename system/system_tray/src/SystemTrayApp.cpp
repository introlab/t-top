#include "SystemTrayApp.h"

SystemTrayApp::SystemTrayApp(int argc, char *argv[])
    :   QGuiApplication(argc, argv)
{
    m_trayIcon = new SystemTrayIcon(this);
    m_trayIcon->show();
}
