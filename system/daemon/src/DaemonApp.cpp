#include "DaemonApp.h"
#include <QtDebug>


DeamonApp::DeamonApp(int argc, char *argv[])
    : QCoreApplication(argc, argv), m_serialManager(nullptr)
{
    qDebug() << "DeamonApp running...";

    // TODO read configuration in a file ? Command line arguments ?
    setupWebSocketServers();
    setupSerialManager();

}

void DeamonApp::setupWebSocketServers()
{
    // Create websocket server for ROS, CLI & TaskBar
    DaemonWebSocketServer *rosServer = new DaemonWebSocketServer("DeamonApp-ROSWebSocketServer", 8080, 1, this);
    DaemonWebSocketServer *cliServer = new DaemonWebSocketServer("DeamonApp-CLIWebSocketServer", 8081, 1, this);
    DaemonWebSocketServer *systemTrayServer = new DaemonWebSocketServer("DeamonApp-SystemTrayWebSocketServer", 8082, 1, this);

    // Add all servers to the list
    m_webSocketServers << rosServer << cliServer << systemTrayServer;
}

void DeamonApp::setupSerialManager()
{
    DaemonSerialManager::printAvailablePorts();

    for (auto &&port : DaemonSerialManager::availablePorts())
    {
        if (port.portName().contains("cu") && port.manufacturer().contains("Teensyduino"))
        {
            qDebug() << "Automatic discovery of port: " << port.portName() << " from manufacturer: " << port.manufacturer();
            m_serialManager = new DaemonSerialManager(port, this);
            break;
        }

    }

    if (!m_serialManager)
    {
        qDebug() << "Automatic discovery of serial port failed.";
    }

}
