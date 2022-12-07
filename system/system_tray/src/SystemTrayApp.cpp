#include "SystemTrayApp.h"

SystemTrayApp::SystemTrayApp(int argc, char* argv[]) : QApplication(argc, argv)
{
    m_trayIcon = new SystemTrayIcon(this);
    m_trayIcon->show();

    // TODO Should get URL from command line args
    QUrl url("ws://localhost:8082");

    m_webSocketSerialManager = new WebSocketSerialManager(url, this);


    // Connect signals
    // TODO remove connect everything for tests...
    connect(m_webSocketSerialManager, &WebSocketSerialManager::newBaseStatus, this, &SystemTrayApp::onNewBaseStatus);
    connect(m_webSocketSerialManager, &WebSocketSerialManager::newButtonPressed, this, &SystemTrayApp::onNewButtonPressed);
    connect(m_webSocketSerialManager, &WebSocketSerialManager::newSetVolume, this, &SystemTrayApp::onNewSetVolume);
    connect(m_webSocketSerialManager, &WebSocketSerialManager::newSetLedColors, this, &SystemTrayApp::onNewSetLedColors);
    connect(m_webSocketSerialManager, &WebSocketSerialManager::newMotorStatus, this, &SystemTrayApp::onNewMotorStatus);
    connect(m_webSocketSerialManager, &WebSocketSerialManager::newImuData, this, &SystemTrayApp::onNewImuData);
    connect(
        m_webSocketSerialManager,
        &WebSocketSerialManager::newSetTorsoOrientation,
        this,
        &SystemTrayApp::onNewSetTorsoOrientation);
    connect(m_webSocketSerialManager, &WebSocketSerialManager::newSetHeadPose, this, &SystemTrayApp::onNewSetHeadPose);
    connect(m_webSocketSerialManager, &WebSocketSerialManager::newShutdown, this, &SystemTrayApp::onNewShutdown);
    connect(m_webSocketSerialManager, &WebSocketSerialManager::newError, this, &SystemTrayApp::onNewError);


}

void SystemTrayApp::onNewBaseStatus(Device source, const BaseStatusPayload &payload)
{
    qDebug() << "********* "
             << "void SystemTrayApp::onNewBaseStatus(Device source, const ButtonPressedPayload &payload)";
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
