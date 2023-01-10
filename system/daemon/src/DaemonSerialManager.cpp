#include "DaemonSerialManager.h"
#include <QDebug>
#include <QDateTime>


DaemonSerialManager::DaemonSerialManager(const QSerialPortInfo& port, QObject* parent)
    : QObject(parent),
      m_serialPort(nullptr)
{
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
    }
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
}

void DaemonSerialManager::onReadyRead()
{
    // Update manager
    qDebug() << "void DaemonSerialManager::onReadyRead()";
    m_serialCommunicationManager->update(QDateTime::currentMSecsSinceEpoch());
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

    // TODO newRoute should not happen on computer device ?
    m_serialCommunicationManager->setRouteCallback([this](Device destination, const uint8_t* data, size_t size)
                                                   { emit this->newRoute(destination, data, size); });

    m_serialCommunicationManager->setSetVolumeHandler([this](Device source, const SetVolumePayload& payload)
                                                      { emit this->newSetVolume(source, payload); });

    m_serialCommunicationManager->setErrorCallback([this](const char* message, tl::optional<MessageType> messageType)
                                                   { emit this->newError(message, messageType); });
}
