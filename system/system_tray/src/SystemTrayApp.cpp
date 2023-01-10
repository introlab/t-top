#include "SystemTrayApp.h"

SystemTrayApp::SystemTrayApp(int argc, char* argv[]) : QApplication(argc, argv)
{
    m_trayIcon = new SystemTrayIcon(this);
    m_trayIcon->show();

    // TODO Should get URL from command line args
    QUrl url("ws://localhost:8082");

    m_webSocketProtocolWrapper = new WebSocketProtocolWrapper(url, this);

    connectWebSocketProtocolWrapperSignals();
    connectSystemTraySignals();
}

void SystemTrayApp::onNewBaseStatus(Device source, const BaseStatusPayload &payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewBaseStatus(Device source, const ButtonPressedPayload &payload)";
    /*
    struct BaseStatusPayload
    {
        static constexpr bool DEFAULT_ACKNOWLEDGMENT_NEEDED = false;
        static constexpr MessageType MESSAGE_TYPE = MessageType::BASE_STATUS;
        static constexpr uint8_t PAYLOAD_SIZE = 42;

         bool isPsuConnected;
         bool hasChargerError;
         bool isBatteryCharging;
         bool hasBatteryError;
         float stateOfCharge;
         float current;
         float voltage;
         float onboardTemperature;
         float externalTemperature;
         float frontLightSensor;
         float backLightSensor;
         float leftLightSensor;
         float rightLightSensor;
         uint8_t volume;
         uint8_t maximumVolume;

          template<class Buffer>
          bool writeTo(Buffer& buffer) const;

        template<class Buffer>
        static tl::optional<BaseStatusPayload> readFrom(Buffer& buffer);

    };
    */
    m_lastBaseStatusPayloadReceived = payload;
    m_trayIcon->enableActions(true);
    m_trayIcon->updateStateOfChargeText(payload.isPsuConnected, payload.hasChargerError, payload.isBatteryCharging,
                                        payload.hasBatteryError, payload.stateOfCharge, payload.current, payload.voltage);

}

void SystemTrayApp::onNewButtonPressed(Device source, const ButtonPressedPayload& payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewButtonPressed(Device source, const ButtonPressedPayload &payload)";
}

void SystemTrayApp::onNewSetVolume(Device source, const SetVolumePayload& payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewSetVolume(Device source, const SetVolumePayload &payload)";
}

void SystemTrayApp::onNewSetLedColors(Device source, const SetLedColorsPayload& payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewSetLedColors(Device source, const SetLedColorsPayload &payload)";
}

void SystemTrayApp::onNewMotorStatus(Device source, const MotorStatusPayload& payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewMotorStatus(Device source, const MotorStatusPayload &payload)";
}

void SystemTrayApp::onNewImuData(Device source, const ImuDataPayload& payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewImuData(Device source, const ImuDataPayload &payload)";
}

void SystemTrayApp::onNewSetTorsoOrientation(Device source, const SetTorsoOrientationPayload& payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewStatus(Device source, const BaseStatusPayload &payload)";
}

void SystemTrayApp::onNewSetHeadPose(Device source, const SetHeadPosePayload& payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewSetHeadPose(Device source, const SetHeadPosePayload &payload)";
}

void SystemTrayApp::onNewShutdown(Device source, const ShutdownPayload& payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewShutdown(Device source, const ShutdownPayload &payload)";
}

void SystemTrayApp::onNewRoute(Device destination, const uint8_t* data, size_t size)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewRoute(Device destination, const uint8_t *data, size_t size)";
}

void SystemTrayApp::onNewError(const char* message, tl::optional<MessageType> messageType)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewError(const char *message, tl::optional<MessageType> messageType)";
    qDebug() << message;
}

void SystemTrayApp::onSystemTrayVolumeUp()
{
    SetVolumePayload payload;
    m_webSocketProtocolWrapper->send(Device::COMPUTER, Device::PSU_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}

void SystemTrayApp::onSystemTrayVolumeDown()
{
    SetVolumePayload payload;
    m_webSocketProtocolWrapper->send(Device::COMPUTER, Device::PSU_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}

void SystemTrayApp::onSystemTrayCloseAllLeds()
{
    SetLedColorsPayload payload;
    m_webSocketProtocolWrapper->send(Device::COMPUTER, Device::PSU_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}

void SystemTrayApp::onSystemTrayResetTorso()
{
    SetTorsoOrientationPayload payload;
    m_webSocketProtocolWrapper->send(Device::COMPUTER, Device::DYNAMIXEL_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}

void SystemTrayApp::onSystemTrayResetHead()
{
    SetHeadPosePayload payload;
    m_webSocketProtocolWrapper->send(Device::COMPUTER, Device::DYNAMIXEL_CONTROL, payload, QDateTime::currentMSecsSinceEpoch());
}


void SystemTrayApp::connectWebSocketProtocolWrapperSignals()
{
    Q_ASSERT(m_webSocketProtocolWrapper);
    // Connect signals
    // TODO remove connect everything for tests...
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newBaseStatus, this, &SystemTrayApp::onNewBaseStatus);
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newButtonPressed, this, &SystemTrayApp::onNewButtonPressed);
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newSetVolume, this, &SystemTrayApp::onNewSetVolume);
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newSetLedColors, this, &SystemTrayApp::onNewSetLedColors);
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newMotorStatus, this, &SystemTrayApp::onNewMotorStatus);
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newImuData, this, &SystemTrayApp::onNewImuData);
    connect(
        m_webSocketProtocolWrapper,
        &WebSocketProtocolWrapper::newSetTorsoOrientation,
        this,
        &SystemTrayApp::onNewSetTorsoOrientation);
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newSetHeadPose, this, &SystemTrayApp::onNewSetHeadPose);
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newShutdown, this, &SystemTrayApp::onNewShutdown);
    connect(m_webSocketProtocolWrapper, &WebSocketProtocolWrapper::newError, this, &SystemTrayApp::onNewError);
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
