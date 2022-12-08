#include "WebSocketSerialManager.h"
#include <QDateTime>
#include "SerialMessages.h"

WebSocketSerialManager::WebSocketSerialManager(const QUrl &url, QObject *parent)
{
    m_webSocketWrapper = new WebSocketSerialWrapper(this);

    m_serialCommunicationManager = std::unique_ptr<SerialCommunicationManager>(new SerialCommunicationManager(
        Device::COMPUTER,
        COMMUNICATION_ACKNOWLEDGMENT_TIMEOUT_MS,
        COMMUNICATION_MAXIMUM_TRIAL_COUNT,
        *m_webSocketWrapper));

    setupSerialCommunicationManagerCallbacks();

    connect(m_webSocketWrapper, &WebSocketSerialWrapper::readyRead, this, &WebSocketSerialManager::onReadyRead);

    m_webSocketWrapper->connectTo(url);

}

void WebSocketSerialManager::onReadyRead()
{
    qDebug() << "void WebSocketSerialManager::onReadyRead()";

    // Update will read / write to websocket
    m_serialCommunicationManager->update(QDateTime::currentMSecsSinceEpoch());
}


void WebSocketSerialManager::setupSerialCommunicationManagerCallbacks()
{
    qDebug() << "WebSocketSerialManager::setupSerialCommunicationManagerCallbacks()";
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

    m_serialCommunicationManager->setRouteCallback([this](Device destination, const uint8_t* data, size_t size)
                                                   { emit this->newRoute(destination, data, size); });

    m_serialCommunicationManager->setSetVolumeHandler([this](Device source, const SetVolumePayload& payload)
                                                      { emit this->newSetVolume(source, payload); });

    m_serialCommunicationManager->setErrorCallback([this](const char* message, tl::optional<MessageType> messageType)
                                                   { emit this->newError(message, messageType); });
}

