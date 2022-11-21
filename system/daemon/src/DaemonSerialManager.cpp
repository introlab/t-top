#include "DaemonSerialManager.h"
#include <QDebug>
#include <QDateTime>


DaemonSerialManager::DaemonSerialManager(const QSerialPortInfo &port, QObject *parent)
    : QObject(parent), m_serialPort(nullptr)
{

    m_serialPort = new DaemonSerialPortWrapper(port, this);

    m_serialCommunicationManager = std::unique_ptr<SerialCommunicationManager>(new SerialCommunicationManager(Device::COMPUTER,
                                                                                                              COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS,
                                                                                                              COMMUNICATION_MAXIMUM_TRIAL_COUNT, *m_serialPort, this));

    // TODO port setup, hardcoded for now
    m_serialPort->setDataBits(QSerialPort::Data8);
    m_serialPort->setStopBits(QSerialPort::OneStop);
    m_serialPort->setBaudRate(QSerialPort::Baud115200, QSerialPort::AllDirections);
    m_serialPort->setFlowControl(QSerialPort::NoFlowControl);

    if (m_serialPort->open(QIODevice::ReadWrite))
    {
        // Connect signals
        connect(m_serialPort, &QSerialPort::errorOccurred, this, &DaemonSerialManager::onErrorOccurred);
        connect(m_serialPort, &QSerialPort::readyRead, this, &DaemonSerialManager::onReadyRead);

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

bool DaemonSerialManager::isValidPort(const QString &name)
{
    for (auto &&port : DaemonSerialManager::availablePorts())
    {
        if(port.portName() == name) {
            return true;
        }
    }

    return false;
}

void DaemonSerialManager::printAvailablePorts()
{
    for (auto &&port : DaemonSerialManager::availablePorts())
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
    m_serialCommunicationManager->setBaseStatusHandler(&DaemonSerialManagerBaseStatusHandler);
    m_serialCommunicationManager->setButtonPressedHandler(&DaemonSerialManagerButtonPressedHandler);
    m_serialCommunicationManager->setMotorStatusHandler(&DaemonSerialManagerMotorStatusHandler);
    m_serialCommunicationManager->setImuDataHandler(&DaemonSerialManagerImuDataHandler);
    m_serialCommunicationManager->setSetHeadPoseHandler(&DaemonSerialManagerSetHeadPoseHandler);
    m_serialCommunicationManager->setSetLedColorsHandler(&DaemonSerialManagerSetLedColorsHandler);
    m_serialCommunicationManager->setSetTorsoOrientationHandler(&DaemonSerialManagerSetTorsoOrientationHandler);
    m_serialCommunicationManager->setShutdownHandler(&DaemonSerialManagerShutdownHandler);
    m_serialCommunicationManager->setRouteCallback(&DaemonSerialManagerRouteCallback);
    m_serialCommunicationManager->setSetVolumeHandler(&DaemonSerialManagerSetVolumeHandler);
    m_serialCommunicationManager->setErrorCallback(&DaemonSerialManagerErrorCallback);
}

void DaemonSerialManagerBaseStatusHandler(Device source, const BaseStatusPayload& payload, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newStatus(source, payload);
    }
}

void DaemonSerialManagerButtonPressedHandler(Device source, const ButtonPressedPayload& payload, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newButtonPressed(source, payload);
    }
}

void DaemonSerialManagerSetVolumeHandler(Device source, const SetVolumePayload& payload, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newSetVolume(source, payload);
    }
}

void DaemonSerialManagerSetLedColorsHandler(Device source, const SetLedColorsPayload& payload, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newSetLedColors(source, payload);
    }
}

void DaemonSerialManagerMotorStatusHandler(Device source, const MotorStatusPayload& payload, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newMotorStatus(source, payload);
    }
}

void DaemonSerialManagerImuDataHandler(Device source, const ImuDataPayload& payload, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newImuData(source, payload);
    }
}

void DaemonSerialManagerSetTorsoOrientationHandler(Device source, const SetTorsoOrientationPayload& payload, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newSetTorsoOrientation(source, payload);
    }
}

void DaemonSerialManagerSetHeadPoseHandler(Device source, const SetHeadPosePayload& payload, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newSetHeadPose(source, payload);
    }
}

void DaemonSerialManagerShutdownHandler(Device source, const ShutdownPayload& payload, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newShutdown(source, payload);
    }
}

void DaemonSerialManagerRouteCallback(Device destination, const uint8_t* data, size_t size, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newRoute(destination, data, size);
    }
}

void DaemonSerialManagerErrorCallback(const char* message, tl::optional<MessageType> messageType, void* userData)
{
    DaemonSerialManager* manager = reinterpret_cast<DaemonSerialManager*>(userData);
    if (manager) {
        emit manager->newError(message, messageType);
    }
}


