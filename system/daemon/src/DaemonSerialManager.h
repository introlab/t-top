#ifndef _DAEMON_SERIAL_MANAGER_H_
#define _DAEMON_SERIAL_MANAGER_H_

#include <QObject>
#include <QSerialPort>
#include <QSerialPortInfo>
#include <QDateTime>
#include <QTimer>
#include "DaemonSerialPortWrapper.h"
#include <memory>
#include <QDebug>

class DaemonSerialManager : public QObject
{
    Q_OBJECT

    DaemonSerialPortWrapper* m_serialPort;
    std::unique_ptr<SerialCommunicationManager> m_serialCommunicationManager;
    QTimer* m_checkSerialPortTimer;

    static constexpr long COMMUNICATION_SERIAL_BAUD_RATE = 115200;
    static constexpr uint32_t COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS = 100;
    static constexpr uint32_t COMMUNICATION_MAXIMUM_TRIAL_COUNT = 5;
    static constexpr int CHECK_SERIAL_PORT_TIMER_INTERVAL_MS = 1000;

public:
    DaemonSerialManager(QObject* parent = nullptr);

    template<class Payload>
    void send(Device destination, const Payload& payload, qint64 timestamp_ms = QDateTime::currentMSecsSinceEpoch());


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
    void newError(const char* message, tl::optional<MessageType> messageType);

private slots:
    void onErrorOccurred(QSerialPort::SerialPortError error);
    void onReadyRead();
    void onTimerCheckSerialPort();

private:
    QSerialPortInfo getAvailableSerialPort();
    void setupSerialCommunicationManagerCallbacks();
};


template<class Payload>
void DaemonSerialManager::send(Device destination, const Payload& payload, qint64 timestamp_ms)
{
    if (m_serialCommunicationManager)
    {
        m_serialCommunicationManager->send(destination, payload, timestamp_ms);
    }
    else
    {
        qWarning() << "DaemonSerialManager::send - cannot send message, serial port not working?";
    }
}

#endif  // _DAEMON_SERIAL_MANAGER_H_
