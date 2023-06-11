#include "DaemonApp.h"
#include <QtDebug>
#include "SerialCommunicationBuffer.h"
#include "SerialMessages.h"
#include "WebSocketProtocolWrapper.h"
#include <QProcess>
#include <QCommandLineParser>
#include "ProcessUtils.h"
#include <cmath>

#include "JetsonModelParser.h"

using namespace std;

DaemonApp::DaemonApp(int& argc, char* argv[])
    : QCoreApplication(argc, argv),
      m_serialManager(nullptr),
      m_lastLightLevel(0.f)
{
    qDebug() << "DaemonApp running...";

    parseJetsonModel();
    setupWebSocketServers();
    setupSerialManager();
}

void DaemonApp::onNewBaseStatus(Device source, const BaseStatusPayload& payload)
{
    setPowerMode(payload.isPsuConnected);
    setScreenBrightness(
        payload.frontLightSensor,
        payload.backLightSensor,
        payload.leftLightSensor,
        payload.rightLightSensor);

    foreach (DaemonWebSocketServer* server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewButtonPressed(Device source, const ButtonPressedPayload& payload)
{
    foreach (DaemonWebSocketServer* server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewMotorStatus(Device source, const MotorStatusPayload& payload)
{
    foreach (DaemonWebSocketServer* server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewImuData(Device source, const ImuDataPayload& payload)
{
    foreach (DaemonWebSocketServer* server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }
}

void DaemonApp::onNewShutdown(Device source, const ShutdownPayload& payload)
{
    foreach (DaemonWebSocketServer* server, m_webSocketServers)
    {
        server->sendToClients(source, payload);
    }

    terminateAllROSProcessesAndShutdown();
}

void DaemonApp::onNewRouteFromWebSocket(Device destination, const uint8_t* data, size_t size)
{
    // Send to serial...
    SerialCommunicationBuffer<SERIAL_COMMUNICATION_BUFFER_SIZE> buffer;
    buffer.write(data, size);
    auto header = *MessageHeader::readFrom(buffer);

    switch (header.messageType())
    {
        case MessageType::SET_VOLUME:
        {
            auto payload = *SetVolumePayload::readFrom(buffer);
            m_serialManager->send(destination, payload);
            break;
        }

        case MessageType::SET_LED_COLORS:
        {
            auto payload = *SetLedColorsPayload::readFrom(buffer);
            m_serialManager->send(destination, payload);
            break;
        }

        case MessageType::SET_TORSO_ORIENTATION:
        {
            auto payload = *SetTorsoOrientationPayload::readFrom(buffer);
            m_serialManager->send(destination, payload);
            break;
        }

        case MessageType::SET_HEAD_POSE:
        {
            auto payload = *SetHeadPosePayload::readFrom(buffer);
            m_serialManager->send(destination, payload);
            break;
        }

        default:
            qWarning() << "DaemonApp::onNewRouteFromWebSocket Message discarded type: " << (int)header.messageType();
            break;
    }
}

void DaemonApp::onNewError(const char* message, tl::optional<MessageType> messageType)
{
    qDebug() << "********* "
             << "void DaemonApp::onNewError(const char *message, tl::optional<MessageType> messageType)";
    qDebug() << message;
}

void DaemonApp::parseJetsonModel()
{
    m_jetsonModel = get_jetson_model();
    qDebug() << "Jetson model: " << get_jetson_model_name(m_jetsonModel).c_str();
}

void DaemonApp::setupWebSocketServers()
{
    // Create websocket server for ROS, CLI & TaskBar
    DaemonWebSocketServer* rosServer = new DaemonWebSocketServer(
        "DaemonApp-ROSWebSocketServer",
        WebSocketProtocolWrapper::ROS_DEFAULT_CLIENT_PORT,
        1,
        this);
    DaemonWebSocketServer* cliServer = new DaemonWebSocketServer(
        "DaemonApp-CLIWebSocketServer",
        WebSocketProtocolWrapper::CLI_DEFAULT_CLIENT_PORT,
        1,
        this);
    DaemonWebSocketServer* systemTrayServer = new DaemonWebSocketServer(
        "DaemonApp-SystemTrayWebSocketServer",
        WebSocketProtocolWrapper::TRAY_DEFAULT_CLIENT_PORT,
        1,
        this);

    // Add all servers to the list
    m_webSocketServers << rosServer << cliServer << systemTrayServer;

    // Connect signals handling messages from websockets to serial port
    foreach (auto server, m_webSocketServers)
    {
        connect(server, &DaemonWebSocketServer::newRoute, this, &DaemonApp::onNewRouteFromWebSocket);
    }
}

void DaemonApp::setupSerialManager()
{
    m_serialManager = new DaemonSerialManager(this);

    // Connect useful signals
    connect(m_serialManager, &DaemonSerialManager::newBaseStatus, this, &DaemonApp::onNewBaseStatus);
    connect(m_serialManager, &DaemonSerialManager::newButtonPressed, this, &DaemonApp::onNewButtonPressed);
    connect(m_serialManager, &DaemonSerialManager::newMotorStatus, this, &DaemonApp::onNewMotorStatus);
    connect(m_serialManager, &DaemonSerialManager::newImuData, this, &DaemonApp::onNewImuData);
    connect(m_serialManager, &DaemonSerialManager::newShutdown, this, &DaemonApp::onNewShutdown);
    connect(m_serialManager, &DaemonSerialManager::newError, this, &DaemonApp::onNewError);
}

void DaemonApp::setPowerMode(bool isPsuConnected)
{
#ifdef __linux__
    if (m_jetsonModel != JetsonModel::ORIN)
    {
        return;
    }

    if ((m_lastIsPsuConnected == tl::nullopt || m_lastIsPsuConnected == true) && !isPsuConnected)
    {
        qDebug() << "********* DaemonApp::setPowerMode Set low power mode";
        QProcess::startDetached("sudo", {"nvpmodel", "-m", LOW_POWER_MODE_MODEL_INDEX});
    }
    else if ((m_lastIsPsuConnected == tl::nullopt || m_lastIsPsuConnected == false) && isPsuConnected)
    {
        qDebug() << "********* DaemonApp::setPowerMode Set high power mode";
        QProcess::startDetached("sudo", {"nvpmodel", "-m", HIGH_POWER_MODE_MODEL_INDEX});
    }

    m_lastIsPsuConnected = isPsuConnected;
#endif
}

void DaemonApp::setScreenBrightness(float front, float back, float left, float right)
{
#ifdef __linux__
    float lightLevel = min(min(front, back), min(left, right));
    m_lastLightLevel = (1.f - LIGHT_LEVEL_ALPHA) * m_lastLightLevel + LIGHT_LEVEL_ALPHA * lightLevel;
    float brightness = MIN_SCREEN_BRIGHTNESS + (MAX_SCREEN_BRIGHTNESS - MIN_SCREEN_BRIGHTNESS) * m_lastLightLevel;

    // TODO change the screen arg
    // xrandr -q | grep " connected" to get connected screen
    // QProcess::startDetached("xrandr", {"--output", "TODO_set_screen", "--brightness", QString::number(brightness)});
#endif
}

void DaemonApp::terminateAllROSProcessesAndShutdown()
{
#ifdef __linux__
    auto pids = listPidsMatchingTheCriteria("roslaunch");
    qDebug() << "Shutdown of roslaunch processes (" << pids << ")";
    shutdownProcessesAndWait(pids, SHUTDOWN_TIMEOUT_SEC);
    qDebug() << "Shutdown of roslaunch processes completed";
    QProcess::startDetached("sudo", {"shutdown", "now"});
#endif
}
