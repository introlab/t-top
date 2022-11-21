#include "DaemonApp.h"
#include <QtDebug>


DaemonApp::DaemonApp(int argc, char *argv[])
    : QCoreApplication(argc, argv), m_serialManager(nullptr)
{
    qDebug() << "DeamonApp running...";

    // TODO read configuration in a file ? Command line arguments ?
    setupWebSocketServers();
    setupSerialManager();

}

void DaemonApp::onNewStatus(Device source, const BaseStatusPayload &payload)
{
    qDebug() << "********* " << "void DeamonApp::onNewStatus(Device source, const BaseStatusPayload &payload)";
}

void DaemonApp::onNewButtonPressed(Device source, const ButtonPressedPayload &payload)
{
    qDebug() << "********* " << "void DeamonApp::onNewButtonPressed(Device source, const ButtonPressedPayload &payload)";
}

void DaemonApp::onNewSetVolume(Device source, const SetVolumePayload &payload)
{
    qDebug() << "********* " << "void DeamonApp::onNewSetVolume(Device source, const SetVolumePayload &payload)";
}

void DaemonApp::onNewSetLedColors(Device source, const SetLedColorsPayload &payload)
{
    qDebug() << "********* " << "void DeamonApp::onNewSetLedColors(Device source, const SetLedColorsPayload &payload)";
}

void DaemonApp::onNewMotorStatus(Device source, const MotorStatusPayload &payload)
{
    qDebug() << "********* " << "void DeamonApp::onNewMotorStatus(Device source, const MotorStatusPayload &payload)";
}

void DaemonApp::onNewImuData(Device source, const ImuDataPayload &payload)
{
    qDebug() << "********* " << "void DeamonApp::onNewImuData(Device source, const ImuDataPayload &payload)";
}

void DaemonApp::onNewSetTorsoOrientation(Device source, const SetTorsoOrientationPayload &payload)
{
    qDebug() << "********* " << "void DeamonApp::onNewStatus(Device source, const BaseStatusPayload &payload)";
}

void DaemonApp::onNewSetHeadPose(Device source, const SetHeadPosePayload &payload)
{
    qDebug() << "********* " << "void DeamonApp::onNewSetHeadPose(Device source, const SetHeadPosePayload &payload)";
}

void DaemonApp::onNewShutdown(Device source, const ShutdownPayload &payload)
{
    qDebug() << "********* " << "void DeamonApp::onNewShutdown(Device source, const ShutdownPayload &payload)";
}

void DaemonApp::onNewRoute(Device destination, const uint8_t *data, size_t size)
{
    qDebug() << "********* " << "void DeamonApp::onNewRoute(Device destination, const uint8_t *data, size_t size)";
}

void DaemonApp::onNewError(const char *message, tl::optional<MessageType> messageType)
{
    qDebug() << "********* " << "void DeamonApp::onNewError(const char *message, tl::optional<MessageType> messageType)";
}

void DaemonApp::setupWebSocketServers()
{
    // Create websocket server for ROS, CLI & TaskBar
    DaemonWebSocketServer *rosServer = new DaemonWebSocketServer("DeamonApp-ROSWebSocketServer", 8080, 1, this);
    DaemonWebSocketServer *cliServer = new DaemonWebSocketServer("DeamonApp-CLIWebSocketServer", 8081, 1, this);
    DaemonWebSocketServer *systemTrayServer = new DaemonWebSocketServer("DeamonApp-SystemTrayWebSocketServer", 8082, 1, this);

    // Add all servers to the list
    m_webSocketServers << rosServer << cliServer << systemTrayServer;
}

void DaemonApp::setupSerialManager()
{
    DaemonSerialManager::printAvailablePorts();

    for (auto &&port : DaemonSerialManager::availablePorts())
    {
        if (port.portName().contains("cu") && port.manufacturer().contains("Teensyduino"))
        {
            qDebug() << "Automatic discovery of port: " << port.portName() << " from manufacturer: " << port.manufacturer();
            m_serialManager = new DaemonSerialManager(port, this);

            //Connect signals
            //TODO remove connect everything for tests...
            connect(m_serialManager, &DaemonSerialManager::newStatus, this, &DaemonApp::onNewStatus);
            connect(m_serialManager, &DaemonSerialManager::newButtonPressed, this, &DaemonApp::onNewButtonPressed);
            connect(m_serialManager, &DaemonSerialManager::newSetVolume, this, &DaemonApp::onNewSetVolume);
            connect(m_serialManager, &DaemonSerialManager::newSetLedColors, this, &DaemonApp::onNewSetLedColors);
            connect(m_serialManager, &DaemonSerialManager::newMotorStatus, this, &DaemonApp::onNewMotorStatus);
            connect(m_serialManager, &DaemonSerialManager::newImuData, this, &DaemonApp::onNewImuData);
            connect(m_serialManager, &DaemonSerialManager::newSetTorsoOrientation, this, &DaemonApp::onNewSetTorsoOrientation);
            connect(m_serialManager, &DaemonSerialManager::newSetHeadPose, this, &DaemonApp::onNewSetHeadPose);
            connect(m_serialManager, &DaemonSerialManager::newShutdown, this, &DaemonApp::onNewShutdown);
            connect(m_serialManager, &DaemonSerialManager::newError, this, &DaemonApp::onNewError);
            break;
        }

    }

    if (!m_serialManager)
    {
        qDebug() << "Automatic discovery of serial port failed.";
    }

}
