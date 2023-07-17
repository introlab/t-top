#include "SystemTrayIcon.h"


SystemTrayIcon::SystemTrayIcon(QObject* parent)
    : QSystemTrayIcon(parent),
      m_greenRobotIcon(":/icons/resources/green-robot-icon.png"),
      m_redRobotIcon(":/icons/resources/red-robot-icon.png")
{
    setToolTip("T-TOP Configuration");
    setIcon(m_redRobotIcon);

    setupMenu();
}

void SystemTrayIcon::setupMenu()
{
    m_menu = new QMenu("T-TOP");

    m_stateOfChargeAction = new QAction("State of charge: ", m_menu);
    m_stateOfChargeAction->setDisabled(true);
    m_volumeUpAction = new QAction("Volume Up", m_menu);
    m_volumeUpAction->setDisabled(true);
    m_volumeDownAction = new QAction("Volume Down", m_menu);
    m_volumeDownAction->setDisabled(true);
    m_closeAllLedsAction = new QAction("Close All Leds", m_menu);
    m_closeAllLedsAction->setDisabled(true);
    m_resetTorsoAction = new QAction("Reset Torso", m_menu);
    m_resetTorsoAction->setDisabled(true);
    m_resetHeadAction = new QAction("Reset Head", m_menu);
    m_resetHeadAction->setDisabled(true);

    m_menu->addAction(m_stateOfChargeAction);
    m_menu->addSeparator();
    m_menu->addAction(m_volumeUpAction);
    m_menu->addAction(m_volumeDownAction);
    m_menu->addSeparator();
    m_menu->addAction(m_closeAllLedsAction);
    m_menu->addAction(m_resetTorsoAction);
    m_menu->addAction(m_resetHeadAction);

    setContextMenu(m_menu);
    setupSignals();
}

void SystemTrayIcon::setupSignals()
{
    // State Of Charge
    connect(m_stateOfChargeAction, &QAction::triggered, this, &SystemTrayIcon::onStateOfChargeAction);

    // Volume Up
    connect(m_volumeUpAction, &QAction::triggered, this, &SystemTrayIcon::onVolumeUpAction);
    connect(m_volumeUpAction, &QAction::triggered, this, &SystemTrayIcon::volumeUpClicked);

    // Volume Down
    connect(m_volumeDownAction, &QAction::triggered, this, &SystemTrayIcon::onVolumeDownAction);
    connect(m_volumeDownAction, &QAction::triggered, this, &SystemTrayIcon::volumeDownClicked);

    // Close All Leds
    connect(m_closeAllLedsAction, &QAction::triggered, this, &SystemTrayIcon::onCloseAllLedsAction);
    connect(m_closeAllLedsAction, &QAction::triggered, this, &SystemTrayIcon::closeAllLedsClicked);

    // Reset Torso
    connect(m_resetTorsoAction, &QAction::triggered, this, &SystemTrayIcon::onResetTorsoAction);
    connect(m_resetTorsoAction, &QAction::triggered, this, &SystemTrayIcon::resetTorsoClicked);

    // Reset Head
    connect(m_resetHeadAction, &QAction::triggered, this, &SystemTrayIcon::onResetHeadAction);
    connect(m_resetHeadAction, &QAction::triggered, this, &SystemTrayIcon::resetHeadClicked);
}

void SystemTrayIcon::enableActions(bool enabled)
{
    m_volumeUpAction->setEnabled(enabled);
    m_volumeDownAction->setEnabled(enabled);
    m_closeAllLedsAction->setEnabled(enabled);
    m_resetTorsoAction->setEnabled(enabled);
    m_resetHeadAction->setEnabled(enabled);
}

void SystemTrayIcon::setConnected(bool connected)
{
    if (connected)
    {
        setIcon(m_greenRobotIcon);
    }
    else
    {
        setIcon(m_redRobotIcon);
        enableActions(false);
    }
}

void SystemTrayIcon::onStateOfChargeAction()
{
    qDebug() << "onStateOfChargeAction";
}

void SystemTrayIcon::onVolumeUpAction()
{
    qDebug() << "onVolumeUpAction";
}

void SystemTrayIcon::onVolumeDownAction()
{
    qDebug() << "onVolumeDownAction";
}

void SystemTrayIcon::onCloseAllLedsAction()
{
    qDebug() << "onCloseAllLedsAction";
}

void SystemTrayIcon::onResetTorsoAction()
{
    qDebug() << "onResetTorsoAction";
}

void SystemTrayIcon::onResetHeadAction()
{
    qDebug() << "onResetHeadAction";
}

void SystemTrayIcon::updateStateOfChargeText(
    bool isPsuConnected,
    bool hasChargerError,
    bool isBatteryCharging,
    bool hasBatteryError,
    float stateOfCharge,
    float current,
    float voltage)
{
    m_stateOfChargeAction->setText(
        QString::number(stateOfCharge, 'f', 1) + " %" + " " + QString::number(voltage, 'f', 1) + " V" + " " +
        QString::number(current, 'f', 1) + " A" + " " + QString(isPsuConnected ? " PSU" : "") + " " +
        QString(isBatteryCharging ? " Charging" : "") + " " + QString(hasChargerError ? " Charger Error" : "") + " " +
        QString(hasBatteryError ? " Battery Error" : ""));
}
