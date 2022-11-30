#ifndef _DAEMON_SERIAL_MANAGER_H_
#define _DAEMON_SERIAL_MANAGER_H_

#include <QObject>
#include <QSerialPort>
#include <QSerialPortInfo>
#include "DaemonSerialPortWrapper.h"
#include <memory>


class DaemonSerialManager : public QObject
{
    Q_OBJECT

public:
    DaemonSerialManager(const QSerialPortInfo& port, QObject* parent = nullptr);

    static QList<QSerialPortInfo> availablePorts();
    static bool isValidPort(const QString& name);
    static void printAvailablePorts();

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
    void onErrorOccurred(QSerialPort::SerialPortError error);
    void onReadyRead();

private:
    DaemonSerialPortWrapper* m_serialPort;
    std::unique_ptr<SerialCommunicationManager> m_serialCommunicationManager;
    void setupSerialCommunicationManagerCallbacks();

    static constexpr long COMMUNICATION_SERIAL_BAUD_RATE = 115200;
    static constexpr uint32_t COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS = 20;
    static constexpr uint32_t COMMUNICATION_MAXIMUM_TRIAL_COUNT = 5;
};


#endif  // _DAEMON_SERIAL_MANAGER_H_
