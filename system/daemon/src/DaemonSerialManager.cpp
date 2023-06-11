#include "DaemonSerialManager.h"
#include <QDebug>
#include <QDateTime>


DaemonSerialManager::DaemonSerialManager(QObject* parent)
    : QObject(parent),
      m_serialPort(nullptr),
      m_checkSerialPortTimer(nullptr)
{
    m_checkSerialPortTimer = new QTimer(this);
    connect(m_checkSerialPortTimer, &QTimer::timeout, this, &DaemonSerialManager::onTimerCheckSerialPort);
    m_checkSerialPortTimer->start(CHECK_SERIAL_PORT_TIMER_INTERVAL_MS);
}

QList<QSerialPortInfo> DaemonSerialManager::availablePorts()
{
    return QSerialPortInfo::availablePorts();
}

bool DaemonSerialManager::isValidPort(const QString& name)
{
    for (auto&& port : DaemonSerialManager::availablePorts())
    {
        if (port.portName() == name)
        {
            return true;
        }
    }

    return false;
}

void DaemonSerialManager::printAvailablePorts()
{
    for (auto&& port : DaemonSerialManager::availablePorts())
    {
        qDebug() << port.portName() << " " << port.manufacturer();
    }
}

void DaemonSerialManager::onErrorOccurred(QSerialPort::SerialPortError error)
{
    qDebug() << "DaemonSerialManager::onErrorOccurred" << error;
    QTimer::singleShot(
        0,
        [this]()
        {
            m_serialCommunicationManager.reset();
            m_serialPort->deleteLater();
            m_serialPort = nullptr;
        });
}

void DaemonSerialManager::onReadyRead()
{
    // Update manager
    m_serialCommunicationManager->update(QDateTime::currentMSecsSinceEpoch());
}

void DaemonSerialManager::onTimerCheckSerialPort()
{
    if (m_serialPort)
    {
        return;
    }

    auto port = getAvailableSerialPort();
    if (port.isNull())
    {
        return;
    }

    m_serialPort = new DaemonSerialPortWrapper(port, this);

    m_serialCommunicationManager = std::unique_ptr<SerialCommunicationManager>(new SerialCommunicationManager(
        Device::COMPUTER,
        COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS,
        COMMUNICATION_MAXIMUM_TRIAL_COUNT,
        *m_serialPort));

    if (m_serialPort->open(QIODevice::ReadWrite))
    {
        // Connect signals
        connect(m_serialPort, &DaemonSerialPortWrapper::errorOccurred, this, &DaemonSerialManager::onErrorOccurred);
        connect(m_serialPort, &DaemonSerialPortWrapper::readyRead, this, &DaemonSerialManager::onReadyRead);

        setupSerialCommunicationManagerCallbacks();
    }
    else
    {
        qDebug() << "Cannot open port: " << port.portName();
        m_serialCommunicationManager.release();
        m_serialPort->deleteLater();
        m_serialPort = nullptr;
    }
}

QSerialPortInfo DaemonSerialManager::getAvailableSerialPort()
{
    // DaemonSerialManager::printAvailablePorts();

    for (auto&& port : QSerialPortInfo::availablePorts())
    {
        // Will accept Teensy board or test ESP32 board at the moment
#if __APPLE__
        if (port.portName().contains("cu") &&
            (port.manufacturer().contains("Teensyduino") || port.manufacturer().contains("Silicon Labs")))
#else
        if (port.portName().contains("tty") &&
            (port.manufacturer().contains("Teensyduino") || port.manufacturer().contains("Silicon Labs")))
#endif
        {
            return port;
        }
    }

    // Need to test isNull.
    return QSerialPortInfo();
}

void DaemonSerialManager::setupSerialCommunicationManagerCallbacks()
{
    qDebug() << "DaemonSerialManager::setupSerialCommunicationManagerCallbacks()";
    m_serialCommunicationManager->setBaseStatusHandler([this](Device source, const BaseStatusPayload& payload)
                                                       { emit this->newBaseStatus(source, payload); });

    m_serialCommunicationManager->setButtonPressedHandler([this](Device source, const ButtonPressedPayload& payload)
                                                          { emit this->newButtonPressed(source, payload); });

    m_serialCommunicationManager->setMotorStatusHandler([this](Device source, const MotorStatusPayload& payload)
                                                        { emit this->newMotorStatus(source, payload); });

    m_serialCommunicationManager->setImuDataHandler([this](Device source, const ImuDataPayload& payload)
                                                    { emit this->newImuData(source, payload); });

    m_serialCommunicationManager->setSetHeadPoseHandler([this](Device source, const SetHeadPosePayload& payload)
                                                        { emit this->newSetHeadPose(source, payload); });

    m_serialCommunicationManager->setSetLedColorsHandler([this](Device source, const SetLedColorsPayload& payload)
                                                         { emit this->newSetLedColors(source, payload); });

    m_serialCommunicationManager->setSetTorsoOrientationHandler(
        [this](Device source, const SetTorsoOrientationPayload& payload)
        { emit this->newSetTorsoOrientation(source, payload); });

    m_serialCommunicationManager->setShutdownHandler([this](Device source, const ShutdownPayload& payload)
                                                     { emit this->newShutdown(source, payload); });

    m_serialCommunicationManager->setRouteCallback(
        [this](Device destination, const uint8_t* data, size_t size)
        { qWarning() << "DaemonSerialManager::setRouteCallback should not be called!"; });

    m_serialCommunicationManager->setSetVolumeHandler([this](Device source, const SetVolumePayload& payload)
                                                      { emit this->newSetVolume(source, payload); });

    m_serialCommunicationManager->setErrorCallback([this](const char* message, tl::optional<MessageType> messageType)
                                                   { emit this->newError(message, messageType); });
}
