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
    static constexpr uint8_t VOLUME_STEP = 3;

public:
    SystemTrayApp(int& argc, char* argv[]);

private slots:

    // From websocket serial manager
    void onNewBaseStatus(Device source, const BaseStatusPayload& payload);
    void onNewError(const char* message, tl::optional<MessageType> messageType);

    void onWebsocketConnected();
    void onWebsocketDisconnected();

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
