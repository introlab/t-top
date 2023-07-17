#ifndef _SYSTEM_TRAY_ICON_H_
#define _SYSTEM_TRAY_ICON_H_

#include <QSystemTrayIcon>
#include <QMenu>
#include <QAction>
#include <QDebug>

/**
 * @brief The SystemTrayIcon class
 * Robot icon from : https://freeicons.io/customer-support-icons/robot-icon-34023, Creative Commons(Attribution 3.0
 * unported)
 *
 */

class SystemTrayIcon : public QSystemTrayIcon
{
    Q_OBJECT

public:
    SystemTrayIcon(QObject* parent = nullptr);
    void setupMenu();
    void setupSignals();

signals:
    void volumeUpClicked();
    void volumeDownClicked();
    void closeAllLedsClicked();
    void resetTorsoClicked();
    void resetHeadClicked();

public slots:
    void updateStateOfChargeText(
        bool isPsuConnected,
        bool hasChargerError,
        bool isBatteryCharging,
        bool hasBatteryError,
        float stateOfCharge,
        float current,
        float voltage);

    void enableActions(bool enabled);
    void setConnected(bool connected);

private slots:
    void onStateOfChargeAction();
    void onVolumeUpAction();
    void onVolumeDownAction();
    void onCloseAllLedsAction();
    void onResetTorsoAction();
    void onResetHeadAction();


private:
    QIcon m_greenRobotIcon;
    QIcon m_redRobotIcon;

    QMenu* m_menu;
    QAction* m_stateOfChargeAction;
    QAction* m_volumeUpAction;
    QAction* m_volumeDownAction;
    QAction* m_closeAllLedsAction;
    QAction* m_resetTorsoAction;
    QAction* m_resetHeadAction;
};

#endif  // _SYSTEM_TRAY_ICON_H_
