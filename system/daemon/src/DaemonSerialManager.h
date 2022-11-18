#ifndef _DAEMON_SERIAL_MANAGER_H_
#define _DAEMON_SERIAL_MANAGER_H_

#include <QObject>
#include <QSerialPort>
#include <QSerialPortInfo>
#include "DaemonSerialPortWrapper.h"


// Callbacks
void DaemonSerialManagerBaseStatusHandler(Device source, const BaseStatusPayload& payload, void* userData);
void DaemonSerialManagerButtonPressedHandler(Device source, const ButtonPressedPayload& payload, void* userData);
void DaemonSerialManagerSetVolumeHandler(Device source, const SetVolumePayload& payload, void* userData);
void DaemonSerialManagerSetLedColorsHandler(Device source, const SetLedColorsPayload& payload, void* userData);
void DaemonSerialManagerMotorStatusHandler(Device source, const MotorStatusPayload& payload, void* userData);
void DaemonSerialManagerImuDataHandler(Device source, const ImuDataPayload& payload, void* userData);
void DaemonSerialManagerSetTorsoOrientationHandler(Device source, const SetTorsoOrientationPayload& payload, void* userData);
void DaemonSerialManagerSetHeadPoseHandler(Device source, const SetHeadPosePayload& payload, void* userData);
void DaemonSerialManagerShutdownHandler(Device source, const ShutdownPayload& payload, void* userData);
void DaemonSerialManagerRouteCallback(Device destination, const uint8_t* data, size_t size, void* userData);
void DaemonSerialManagerErrorCallback(const char* message, tl::optional<MessageType> messageType, void* userData);

class DaemonSerialManager : public QObject {

    // Callbacks
    friend void DaemonSerialManagerBaseStatusHandler(Device source, const BaseStatusPayload& payload, void* userData);
    friend void DaemonSerialManagerButtonPressedHandler(Device source, const ButtonPressedPayload& payload, void* userData);
    friend void DaemonSerialManagerSetVolumeHandler(Device source, const SetVolumePayload& payload, void* userData);
    friend void DaemonSerialManagerSetLedColorsHandler(Device source, const SetLedColorsPayload& payload, void* userData);
    friend void DaemonSerialManagerMotorStatusHandler(Device source, const MotorStatusPayload& payload, void* userData);
    friend void DaemonSerialManagerImuDataHandler(Device source, const ImuDataPayload& payload, void* userData);
    friend void DaemonSerialManagerSetTorsoOrientationHandler(Device source, const SetTorsoOrientationPayload& payload, void* userData);
    friend void DaemonSerialManagerSetHeadPoseHandler(Device source, const SetHeadPosePayload& payload, void* userData);
    friend void DaemonSerialManagerShutdownHandler(Device source, const ShutdownPayload& payload, void* userData);
    friend void DaemonSerialManagerRouteCallback(Device destination, const uint8_t* data, size_t size, void* userData);
    friend void DaemonSerialManagerErrorCallback(const char* message, tl::optional<MessageType> messageType, void* userData);

    Q_OBJECT

public:

    DaemonSerialManager(const QSerialPortInfo &port, QObject *parent=nullptr);

    static QList<QSerialPortInfo> availablePorts();
    static bool isValidPort(const QString &name);
    static void printAvailablePorts();


private slots:
     void onErrorOccurred(QSerialPort::SerialPortError error);
     void onReadyRead();

private:
    DaemonSerialPortWrapper *m_serialPort;
    SerialCommunicationManager *m_serialCommunicationManager;
    void setupSerialCommunicationManagerCallbacks();

    static constexpr long COMMUNICATION_SERIAL_BAUD_RATE = 115200;
    static constexpr uint32_t COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS = 20;
    static constexpr uint32_t COMMUNICATION_MAXIMUM_TRIAL_COUNT = 5;

};


#endif // _DAEMON_SERIAL_MANAGER_H_
