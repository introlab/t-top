#ifndef _SYSTEM_TRAY_APP_H_
#define _SYSTEM_TRAY_APP_H_

#include <QApplication>
#include <QObject>
#include "SystemTrayIcon.h"
#include "WebSocketProtocolWrapper.h"
#include <memory>

class SystemTrayApp : public QApplication
{
    Q_OBJECT
public:
    SystemTrayApp(int &argc, char* argv[]);

private slots:

    // From websocket serial manager
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

    // From system tray menu
    void onSystemTrayVolumeUp();
    void onSystemTrayVolumeDown();
    void onSystemTrayCloseAllLeds();
    void onSystemTrayResetTorso();
    void onSystemTrayResetHead();

private:

    void connectWebSocketProtocolWrapperSignals();
    void connectSystemTraySignals();


    SystemTrayIcon* m_trayIcon;
    WebSocketProtocolWrapper* m_webSocketProtocolWrapper;

    BaseStatusPayload m_lastBaseStatusPayloadReceived;
};

#endif  // _SYSTEM_TRAY_APP_H_
