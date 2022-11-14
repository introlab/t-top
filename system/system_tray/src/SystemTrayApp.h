#ifndef _SYSTEM_TRAY_APP_H_
#define _SYSTEM_TRAY_APP_H_

#include <QApplication>
#include <QObject>
#include "SystemTrayIcon.h"

class SystemTrayApp : public QApplication
{
    Q_OBJECT
public:
    SystemTrayApp(int argc, char* argv[]);

private:
    SystemTrayIcon *m_trayIcon;
};

#endif // _SYSTEM_TRAY_APP_H_
