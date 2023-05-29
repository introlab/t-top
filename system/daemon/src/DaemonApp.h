#ifndef _DAEMON_APP_H_
#define _DAEMON_APP_H_

#include <QCoreApplication>
#include <QWebSocketServer>
#include <QWebSocket>
#include <QTimer>
#include <QList>
#include "DaemonWebSocketServer.h"
#include "DaemonSerialManager.h"
#include "JetsonModelParser.h"


constexpr const char* LOW_POWER_MODE_MODEL_INDEX = "0";
constexpr const char* HIGH_POWER_MODE_MODEL_INDEX = "1";

constexpr float MIN_SCREEN_BRIGHTNESS = 0.5f;
constexpr float MAX_SCREEN_BRIGHTNESS = 1.f;
constexpr float LIGHT_LEVEL_ALPHA = 0.3;

constexpr qint64 SHUTDOWN_TIMEOUT_SEC = 90;

class DaemonApp : public QCoreApplication
{
    Q_OBJECT

public:
    DaemonApp(int& argc, char* argv[]);

private slots:

    // Serial port events
    void onNewBaseStatus(Device source, const BaseStatusPayload& payload);
    void onNewButtonPressed(Device source, const ButtonPressedPayload& payload);
    void onNewMotorStatus(Device source, const MotorStatusPayload& payload);
    void onNewImuData(Device source, const ImuDataPayload& payload);
    void onNewShutdown(Device source, const ShutdownPayload& payload);
    void onNewError(const char* message, tl::optional<MessageType> messageType);

    // Websocket events
    void onNewRouteFromWebSocket(Device destination, const uint8_t* data, size_t size);

private:
    void parseJetsonModel();

    void setupWebSocketServers();
    void setupSerialManager();
    void setPowerMode(bool isPsuConnected);
    void setScreenBrightness(float front, float back, float left, float right);
    void terminateAllROSProcessesAndShutdown();

    JetsonModel m_jetsonModel;
    tl::optional<bool> m_lastIsPsuConnected;

    float m_lastLightLevel;

    QList<DaemonWebSocketServer*> m_webSocketServers;
    DaemonSerialManager* m_serialManager;
};

#endif  //_DAEMON_APP_H_
