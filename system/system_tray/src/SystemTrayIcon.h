#ifndef _SYSTEM_TRAY_ICON_H_
#define _SYSTEM_TRAY_ICON_H_


#include <QSystemTrayIcon>

//:/icons/resources/robot-icon.png

class SystemTrayIcon : public QSystemTrayIcon {

    Q_OBJECT

public:
    SystemTrayIcon(QObject *parent=nullptr)
        : QSystemTrayIcon(QIcon(":/icons/resources/robot-icon.png"), parent)
    {
        setToolTip("TTOP Configuration");
    }

};


#endif // SYSTEMTRAYICON_H
