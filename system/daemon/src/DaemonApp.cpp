#include "DaemonApp.h"
#include <QtDebug>


DeamonApp::DeamonApp(int argc, char *argv[])
    : QCoreApplication(argc, argv)
{
    qDebug() << "DeamonApp running...";
    create_websocket_server(8080);
}
