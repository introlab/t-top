#include "SystemTrayApp.h"

SystemTrayApp::SystemTrayApp(int& argc, char* argv[]) : QApplication(argc, argv)
{
    m_trayIcon = new SystemTrayIcon(this);
    m_trayIcon->show();

    m_webSocketProtocolWrapper =
        new WebSocketProtocolWrapper(QUrl(WebSocketProtocolWrapper::TRAY_DEFAULT_CLIENT_URL), this);

    connectWebSocketProtocolWrapperSignals();
    connectSystemTraySignals();
}

void SystemTrayApp::onNewBaseStatus(Device source, const BaseStatusPayload& payload)
{
    m_lastBaseStatusPayloadReceived = payload;
    m_trayIcon->enableActions(true);
    m_trayIcon->updateStateOfChargeText(
        payload.isPsuConnected,
        payload.hasChargerError,
        payload.isBatteryCharging,
        payload.hasBatteryError,
        payload.stateOfCharge,
        payload.current,
        payload.voltage);
}

void SystemTrayApp::onNewError(const char* message, tl::optional<MessageType> messageType)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewError(const char *message, tl::optional<MessageType> messageType)";
    qDebug() << message;
}

void SystemTrayApp::onWebsocketConnected()
{
    m_trayIcon->setConnected(true);
}

void SystemTrayApp::onWebsocketDisconnected()
{
    m_trayIcon->setConnected(false);
}

void SystemTrayApp::onSystemTrayVolumeUp()
{
    SetVolumePayload payload;
    payload.volume = m_lastBaseStatusPayloadReceived.volume + VOLUME_STEP;
    m_webSocketProtocolWrapper
        ->send(Device::COMPUTER, Device::PSU_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}

void SystemTrayApp::onSystemTrayVolumeDown()
{
    SetVolumePayload payload;
    if (m_lastBaseStatusPayloadReceived.volume > VOLUME_STEP)
    {
        payload.volume = m_lastBaseStatusPayloadReceived.volume - VOLUME_STEP;
    }
    else
    {
        payload.volume = 0;
    }

    m_webSocketProtocolWrapper
        ->send(Device::COMPUTER, Device::PSU_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}

void SystemTrayApp::onSystemTrayCloseAllLeds()
{
    SetLedColorsPayload payload;
    for (auto i = 0; i < SetLedColorsPayload::LED_COUNT; i++)
    {
        payload.colors[i] = Color{0, 0, 0};
    }

    m_webSocketProtocolWrapper
        ->send(Device::COMPUTER, Device::PSU_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}

void SystemTrayApp::onSystemTrayResetTorso()
{
    SetTorsoOrientationPayload payload;
    payload.torsoOrientation = 0.f;
    m_webSocketProtocolWrapper
        ->send(Device::COMPUTER, Device::DYNAMIXEL_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}

void SystemTrayApp::onSystemTrayResetHead()
{
    SetHeadPosePayload payload;
    payload.headPosePositionX = 0.f;
    payload.headPosePositionY = 0.f;
    payload.headPosePositionZ = 0.17029453895440808f;  // Z height when motors set to 0 according to CAD.
    payload.headPoseOrientationW = 1.f;
    payload.headPoseOrientationX = 0.f;
    payload.headPoseOrientationY = 0.f;
    payload.headPoseOrientationZ = 0.f;
    m_webSocketProtocolWrapper
        ->send(Device::COMPUTER, Device::DYNAMIXEL_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}


void SystemTrayApp::connectWebSocketProtocolWrapperSignals()
{
    Q_ASSERT(m_webSocketProtocolWrapper);
    // Connect signals
    connect(
        m_webSocketProtocolWrapper,
        &WebSocketProtocolWrapper::newBaseStatus,
        this,
        &SystemTrayApp::onNewBaseStatus);
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newError, this, &SystemTrayApp::onNewError);
    connect(
        m_webSocketProtocolWrapper,
        &WebSocketProtocolWrapper::connected,
        this,
        &SystemTrayApp::onWebsocketConnected);
    connect(
        m_webSocketProtocolWrapper,
        &WebSocketProtocolWrapper::disconnected,
        this,
        &SystemTrayApp::onWebsocketDisconnected);
}

void SystemTrayApp::connectSystemTraySignals()
{
    Q_ASSERT(m_trayIcon);
    connect(m_trayIcon, &SystemTrayIcon::volumeUpClicked, this, &SystemTrayApp::onSystemTrayVolumeUp);
    connect(m_trayIcon, &SystemTrayIcon::volumeDownClicked, this, &SystemTrayApp::onSystemTrayVolumeDown);
    connect(m_trayIcon, &SystemTrayIcon::closeAllLedsClicked, this, &SystemTrayApp::onSystemTrayCloseAllLeds);
    connect(m_trayIcon, &SystemTrayIcon::resetTorsoClicked, this, &SystemTrayApp::onSystemTrayResetTorso);
    connect(m_trayIcon, &SystemTrayIcon::resetHeadClicked, this, &SystemTrayApp::onSystemTrayResetHead);
}
