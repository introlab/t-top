#ifndef _WEBSOCKET_SERIAL_MANAGER_H_
#define _WEBSOCKET_SERIAL_MANAGER_H_

#include "SerialCommunicationManager.h"
#include "WebSocketSerialWrapper.h"
#include <QObject>


class WebSocketSerialManager : public QObject
{
    Q_OBJECT

public:
    WebSocketSerialManager(const QUrl &url, QObject* parent = nullptr);


signals:
    void newBaseStatus(Device source, const BaseStatusPayload& payload);
    void newButtonPressed(Device source, const ButtonPressedPayload& payload);
    void newSetVolume(Device source, const SetVolumePayload& payload);
    void newSetLedColors(Device source, const SetLedColorsPayload& payload);
    void newMotorStatus(Device source, const MotorStatusPayload& payload);
    void newImuData(Device source, const ImuDataPayload& payload);
    void newSetTorsoOrientation(Device source, const SetTorsoOrientationPayload& payload);
    void newSetHeadPose(Device source, const SetHeadPosePayload& payload);
    void newShutdown(Device source, const ShutdownPayload& payload);
    void newRoute(Device destination, const uint8_t* data, size_t size);
    void newError(const char* message, tl::optional<MessageType> messageType);

private slots:

    void onReadyRead();

private:
    WebSocketSerialWrapper* m_webSocketWrapper;
    std::unique_ptr<SerialCommunicationManager> m_serialCommunicationManager;
    void setupSerialCommunicationManagerCallbacks();

    static constexpr long COMMUNICATION_SERIAL_BAUD_RATE = 115200;
    static constexpr uint32_t COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS = 20;
    static constexpr uint32_t COMMUNICATION_MAXIMUM_TRIAL_COUNT = 5;
};


#endif // _WEBSOCKET_SERIAL_MANAGER_H_
