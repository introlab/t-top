#include "SystemTrayIcon.h"


SystemTrayIcon::SystemTrayIcon(QObject* parent)
    : QSystemTrayIcon(QIcon(":/icons/resources/robot-icon.png"), parent),
      m_menu(nullptr)
{
    setToolTip("TTOP Configuration");
    setupMenu();
}

void SystemTrayIcon::setupMenu()
{
    m_menu = new QMenu("TTOP");


    m_stateOfChargeAction = new QAction("State of charge: ", m_menu);
    m_volumeUpAction = new QAction("Volume Up", m_menu);
    m_volumeDownAction = new QAction("Volume Down", m_menu);
    m_closeAllLedsAction = new QAction("Close All Leds", m_menu);
    m_resetTorsoAction = new QAction("Reset Torso", m_menu);
    m_resetHeadAction = new QAction("Reset Head", m_menu);

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
