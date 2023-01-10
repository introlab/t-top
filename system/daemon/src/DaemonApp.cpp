#include "DaemonApp.h"
#include <QtDebug>
#include "SerialCommunicationBuffer.h"
#include "SerialMessages.h"

DaemonApp::DaemonApp(int argc, char* argv[]) : QCoreApplication(argc, argv), m_serialManager(nullptr)
{
    qDebug() << "DaemonApp running...";

    // TODO read configuration in a file ? Command line arguments ?
    setupWebSocketServers();
    setupSerialManager();
}

void DaemonApp::onNewBaseStatus(Device source, const BaseStatusPayload& payload)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewBaseStatus(Device source, const BaseStatusPayload &payload)";

    foreach (DaemonWebSocketServer *server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewButtonPressed(Device source, const ButtonPressedPayload& payload)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewButtonPressed(Device source, const ButtonPressedPayload &payload)";
    foreach (DaemonWebSocketServer *server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewSetVolume(Device source, const SetVolumePayload& payload)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewSetVolume(Device source, const SetVolumePayload &payload)";
    foreach (DaemonWebSocketServer *server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewSetLedColors(Device source, const SetLedColorsPayload& payload)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewSetLedColors(Device source, const SetLedColorsPayload &payload)";
    foreach (DaemonWebSocketServer *server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewMotorStatus(Device source, const MotorStatusPayload& payload)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewMotorStatus(Device source, const MotorStatusPayload &payload)";
    foreach (DaemonWebSocketServer *server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewImuData(Device source, const ImuDataPayload& payload)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewImuData(Device source, const ImuDataPayload &payload)";
    foreach (DaemonWebSocketServer *server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewSetTorsoOrientation(Device source, const SetTorsoOrientationPayload& payload)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewStatus(Device source, const BaseStatusPayload &payload)";
    foreach (DaemonWebSocketServer *server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewSetHeadPose(Device source, const SetHeadPosePayload& payload)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewSetHeadPose(Device source, const SetHeadPosePayload &payload)";
    foreach (DaemonWebSocketServer *server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewShutdown(Device source, const ShutdownPayload& payload)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewShutdown(Device source, const ShutdownPayload &payload)";
    foreach (DaemonWebSocketServer *server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewRoute(Device destination, const uint8_t* data, size_t size)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewRoute(Device destination, const uint8_t *data, size_t size)";
}

void DaemonApp::onNewError(const char* message, tl::optional<MessageType> messageType)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewError(const char *message, tl::optional<MessageType> messageType)";
    qDebug() << message;
}

void DaemonApp::setupWebSocketServers()
{
    // Create websocket server for ROS, CLI & TaskBar
    DaemonWebSocketServer* rosServer = new DaemonWebSocketServer("DaemonApp-ROSWebSocketServer", 8080, 1, this);
    DaemonWebSocketServer* cliServer = new DaemonWebSocketServer("DaemonApp-CLIWebSocketServer", 8081, 1, this);
    DaemonWebSocketServer* systemTrayServer =
        new DaemonWebSocketServer("DaemonApp-SystemTrayWebSocketServer", 8082, 1, this);

    // Add all servers to the list
    m_webSocketServers << rosServer << cliServer << systemTrayServer;
}

void DaemonApp::setupSerialManager()
{
    DaemonSerialManager::printAvailablePorts();

    for (auto&& port : DaemonSerialManager::availablePorts())
    {
        //Will accept Teensy board or test ESP32 board at the moment
#if __APPLE__
        if (port.portName().contains("cu") && (port.manufacturer().contains("Teensyduino") || port.manufacturer().contains("Silicon Labs")))
#else
    if (port.portName().contains("tty") && (port.manufacturer().contains("Teensyduino") || port.manufacturer().contains("Silicon Labs")))
#endif
        {
            qDebug() << "Automatic discovery of port: " << port.portName()
                     << " from manufacturer: " << port.manufacturer();
            m_serialManager = new DaemonSerialManager(port, this);

            // Connect signals
            // TODO remove connect everything for tests...
            connect(m_serialManager, &DaemonSerialManager::newBaseStatus, this, &DaemonApp::onNewBaseStatus);
            connect(m_serialManager, &DaemonSerialManager::newButtonPressed, this, &DaemonApp::onNewButtonPressed);
            connect(m_serialManager, &DaemonSerialManager::newSetVolume, this, &DaemonApp::onNewSetVolume);
            connect(m_serialManager, &DaemonSerialManager::newSetLedColors, this, &DaemonApp::onNewSetLedColors);
            connect(m_serialManager, &DaemonSerialManager::newMotorStatus, this, &DaemonApp::onNewMotorStatus);
            connect(m_serialManager, &DaemonSerialManager::newImuData, this, &DaemonApp::onNewImuData);
            connect(
                m_serialManager,
                &DaemonSerialManager::newSetTorsoOrientation,
                this,
                &DaemonApp::onNewSetTorsoOrientation);
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
