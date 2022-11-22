#ifndef _DAEMON_APP_H_
#define _DAEMON_APP_H_

#include <QCoreApplication>
#include <QWebSocketServer>
#include <QWebSocket>
#include <QTimer>
#include <QList>
#include "DaemonWebSocketServer.h"
#include "DaemonSerialManager.h"


class DaemonApp : public QCoreApplication
{
    Q_OBJECT

public:
    DaemonApp(int argc, char* argv[]);

private slots:

    void onNewBaseStatus(Device source, const BaseStatusPayload& payload);
    void onNewButtonPressed(Device source, const ButtonPressedPayload& payload);
    void onNewSetVolume(Device source, const SetVolumePayload& payload);
    void onNewSetLedColors(Device source, const SetLedColorsPayload& payload);
    void onNewMotorStatus(Device source, const MotorStatusPayload& payload);
    void onNewImuData(Device source, const ImuDataPayload& payload);
    void onNewSetTorsoOrientation(Device source, const SetTorsoOrientationPayload& payload);
    void onNewSetHeadPose(Device source, const SetHeadPosePayload& payload);
    void onNewShutdown(Device source, const ShutdownPayload& payload);
    void onNewRoute(Device destination, const uint8_t* data, size_t size);
    void onNewError(const char* message, tl::optional<MessageType> messageType);

private:
    void setupWebSocketServers();
    void setupSerialManager();

    QList<DaemonWebSocketServer*> m_webSocketServers;
    DaemonSerialManager* m_serialManager;
};

#endif  //_DAEMON_APP_H_
