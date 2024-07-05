#include "Connect4Widget.h"
#include "QtUtils.h"

#include <t_top_hbba_lite/Desires.h>

#include <QVBoxLayout>
#include <QUrl>
#include <QUrlQuery>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaEnum>

constexpr float ENABLED_VOLUME = 1;
constexpr float DISABLED_VOLUME = 0;
constexpr int SET_VOLUME_TIMER_INTERVAL_MS = 500;
constexpr int WEB_SOCKET_STATUS_TIMEOUT_MS = 1000;


Connect4Widget::Connect4Widget(rclcpp::Node::SharedPtr node, std::shared_ptr<DesireSet> desireSet, QWidget* parent)
    : m_node(std::move(node)),
      m_desireSet(std::move(desireSet)),
      m_enabled(false),
      m_connect4ManagerConnectionRequested(false)
{
    m_imageDisplay = new ImageDisplay;
    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(m_imageDisplay);
    setLayout(layout);

    m_startButtonPressedSub = m_node->create_subscription<std_msgs::msg::Empty>(
        "daemon/start_button_pressed",
        1,
        [this](const std_msgs::msg::Empty::SharedPtr msg) { startButtonPressedCallback(msg); });
    m_stopButtonPressedSub =  m_node->create_subscription<std_msgs::msg::Empty>(
        "daemon/stop_button_pressed",
        1,
        [this](const std_msgs::msg::Empty::SharedPtr msg) { stopButtonPressedCallback(msg); });

    m_remoteImageSub =  m_node->create_subscription<opentera_webrtc_ros_msgs::msg::PeerImage>(
        "webrtc_image",
        1,
        [this](const opentera_webrtc_ros_msgs::msg::PeerImage::SharedPtr msg) { remoteImageCallback(msg); });

    m_openteraEventSubscriber = m_node->create_subscription<opentera_webrtc_ros_msgs::msg::OpenTeraEvent>(
        "events",
        1,
        [this](const opentera_webrtc_ros_msgs::msg::OpenTeraEvent::SharedPtr msg) { openteraEventCallback(msg); });



    m_volumePub = m_node->create_publisher<std_msgs::msg::Float32>("volume", 1);

    m_setVolumeTimer = new QTimer(this);
    connect(m_setVolumeTimer, &QTimer::timeout, this, &Connect4Widget::onSetVolumeTimerTimeout);
    m_setVolumeTimer->start(SET_VOLUME_TIMER_INTERVAL_MS);

    m_connect4ManagerWebSocketTimer = new QTimer(this);
    connect(m_connect4ManagerWebSocketTimer, &QTimer::timeout, this, &Connect4Widget::onConnect4ManagerWebSocketTimeout);
    m_connect4ManagerWebSocketTimer->start(WEB_SOCKET_STATUS_TIMEOUT_MS);

    m_connect4ManagerWebSocket = new QWebSocket(QString(), QWebSocketProtocol::VersionLatest, this);
    connect(m_connect4ManagerWebSocket, &QWebSocket::sslErrors, this, &Connect4Widget::onConnect4ManagerWebSocketSslErrors);
    connect(m_connect4ManagerWebSocket, &QWebSocket::connected, this, &Connect4Widget::onConnect4ManagerWebSocketConnected);
    connect(m_connect4ManagerWebSocket, &QWebSocket::disconnected, this, &Connect4Widget::onConnect4ManagerWebSocketDisconnected);
    connect(m_connect4ManagerWebSocket, QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::error), this, &Connect4Widget::onConnect4ManagerWebSocketErrorOccurred);
    connect(m_connect4ManagerWebSocket, &QWebSocket::textMessageReceived, this, &Connect4Widget::onConnect4ManagerWebSocketTextMessageReceived);
}

void Connect4Widget::onSetVolumeTimerTimeout()
{
    if (m_enabled)
    {
        setVolume(ENABLED_VOLUME);
    }
    else
    {
        setVolume(DISABLED_VOLUME);
    }
}

void Connect4Widget::onConnect4ManagerWebSocketTimeout()
{
    if (!m_connect4ManagerConnectionRequested)
    {
        return;
    }

    if (m_connect4ManagerWebSocket->state() == QAbstractSocket::UnconnectedState)
    {
        m_connect4ManagerWebSocket->open(m_connect4ManagerWebSocketUrl);
    }
    else
    {
        m_connect4ManagerWebSocket->ping();
    }
}

void Connect4Widget::onConnect4ManagerWebSocketSslErrors(const QList<QSslError>& errors)
{
    QString errorString;
    for (auto& error : errors)
    {
        errorString += error.errorString() + ", ";
    }
    RCLCPP_ERROR_STREAM(m_node->get_logger(), "Connect 4 manager web socket SSL errors: " << errorString.toStdString());
}

void Connect4Widget::onConnect4ManagerWebSocketConnected()
{
    RCLCPP_INFO(m_node->get_logger(), "Connect 4 manager web socket connected");

    QJsonObject dataObject;
    dataObject.insert("participant_name", m_observedParticipantName);
    sendConnect4ManagerEvent("add_observer", dataObject);
}

void Connect4Widget::onConnect4ManagerWebSocketDisconnected()
{
    RCLCPP_INFO(m_node->get_logger(), "Connect 4 manager web socket disconnected");
}

void Connect4Widget::onConnect4ManagerWebSocketErrorOccurred(QAbstractSocket::SocketError error)
{
    RCLCPP_ERROR_STREAM(m_node->get_logger(), "Connect 4 manager web socket error: " <<
        QMetaEnum::fromType<QAbstractSocket::SocketError>().valueToKey(error));
}

void Connect4Widget::onConnect4ManagerWebSocketTextMessageReceived(const QString& message)
{
    RCLCPP_INFO_STREAM(m_node->get_logger(), "Connect 4 manager web socket message: " << message.toStdString());

    QJsonParseError jsonParseError;
    QJsonDocument jsonMessage = QJsonDocument::fromJson(message.toUtf8(), &jsonParseError);
    if (jsonParseError.error != QJsonParseError::NoError)
    {
        RCLCPP_ERROR_STREAM(m_node->get_logger(), "Connect 4 manager web socket message parsing error: " <<
            jsonParseError.errorString().toStdString());
        return;
    }

    if (jsonMessage["event"] == "game_finished")
    {
        handleGameFinishedEvent(jsonMessage["data"]["result"].toString());
    }
}

void Connect4Widget::startButtonPressedCallback(const std_msgs::msg::Empty::SharedPtr msg)
{
    m_enabled = true;
    setVolume(ENABLED_VOLUME);
    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire<NearestFaceFollowingDesire>();
    m_desireSet->addDesire<Camera3dRecordingDesire>();
}

void Connect4Widget::stopButtonPressedCallback(const std_msgs::msg::Empty::SharedPtr msg)
{
    m_enabled = false;
    setVolume(DISABLED_VOLUME);
    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->removeAllDesiresOfType<NearestFaceFollowingDesire>();
    m_desireSet->removeAllDesiresOfType<Camera3dRecordingDesire>();
    m_desireSet->removeAllDesiresOfType<LedAnimationDesire>();
    invokeLater([this]() { m_imageDisplay->setImage(QImage()); });
}

void Connect4Widget::remoteImageCallback(const opentera_webrtc_ros_msgs::msg::PeerImage::SharedPtr msg)
{
    if (msg->frame.encoding != "bgr8")
    {
        RCLCPP_ERROR(m_node->get_logger(), "Not support image encoding (Connect4Widget::remoteImageCallback).");
        return;
    }

    if (m_enabled)
    {
        invokeLater(
            [this, msg]()
            {
                QImage image(&msg->frame.data[0], msg->frame.width, msg->frame.height, QImage::Format_RGB888);
                m_imageDisplay->setImage(image.rgbSwapped());
            });
    }
}

void Connect4Widget::setVolume(float volume)
{
    std_msgs::msg::Float32 msg;
    msg.data = volume;
    m_volumePub->publish(msg);
}

void Connect4Widget::openteraEventCallback(const opentera_webrtc_ros_msgs::msg::OpenTeraEvent::SharedPtr msg)
{
    if (!msg->join_session_events.empty())
    {
        std::string deviceName = msg->current_device_name;
        std::string sessionUrl = msg->join_session_events[0].session_url;
        std::string sessionParameters = msg->join_session_events[0].session_parameters;

        invokeLater(
            [=]()
            {
                m_connect4ManagerConnectionRequested = true;

                parseSessionUrl(sessionUrl, m_connect4ManagerWebSocketUrl, m_connect4ManagerWebSocketPassword);
                m_observedParticipantName = getParticipantName(deviceName, sessionParameters);

                RCLCPP_INFO_STREAM(m_node->get_logger(), "Connect4 Manager Web Socket: Connection (sessionUrl=" << sessionUrl <<
                    ", webSocketUrl=" << m_connect4ManagerWebSocketUrl.toStdString() <<
                    ", password=" << m_connect4ManagerWebSocketPassword.toStdString() <<
                    ", deviceName=" << deviceName <<
                    ", participantName" << m_observedParticipantName.toStdString() << ")");

                m_connect4ManagerWebSocket->open(m_connect4ManagerWebSocketUrl);
            });
    }
    if (!msg->stop_session_events.empty())
    {
        invokeLater(
            [this]()
            {
                m_connect4ManagerConnectionRequested = false;
                m_connect4ManagerWebSocketUrl = "";
                m_connect4ManagerWebSocketPassword = "";
                m_observedParticipantName = "";

                m_connect4ManagerWebSocket->close();
            });
    }
}

bool Connect4Widget::sendConnect4ManagerEvent(const QString& event, const QJsonObject& data)
{
    if (!m_connect4ManagerWebSocket->isValid())
    {
        return false;
    }

    QJsonObject eventObject;
    eventObject.insert("event", event);
    eventObject.insert("data", data);

    QJsonDocument document;
    document.setObject(eventObject);
    QByteArray bytes = document.toJson();
    return m_connect4ManagerWebSocket->sendTextMessage(bytes) == bytes.size();
}

QString Connect4Widget::getParticipantName(const std::string& deviceName, const std::string& sessionParameters)
{
    QJsonParseError jsonParseError;
    QJsonDocument jsonMessage = QJsonDocument::fromJson(QString(sessionParameters.c_str()).toUtf8(), &jsonParseError);

    if (jsonParseError.error != QJsonParseError::NoError)
    {
        RCLCPP_ERROR_STREAM(m_node->get_logger(), "Connect4 Game Data Channel: getParticipantName error: " <<
            jsonParseError.errorString().toStdString());
        return "";
    }

    QJsonObject jsonObject = jsonMessage.object();
    QString participant1 = jsonObject["participant1"].toString();
    QString participant2 = jsonObject["participant2"].toString();
    QString robot1 = jsonObject["robot1"].toString();
    QString robot2 = jsonObject["robot2"].toString();

    QString qDeviceName(deviceName.c_str());
    if (qDeviceName == robot1)
    {
        return participant1;
    }
    else if (qDeviceName == robot2)
    {
        return participant2;
    }
    else
    {
        RCLCPP_ERROR(m_node->get_logger(), "Connect4 Game Data Channel: Device name not found");
        return "";
    }
}

void Connect4Widget::parseSessionUrl(const std::string& sessionUrl, QString& webSocketUrl, QString& password)
{
    std::string baseUrl = sessionUrl.substr(0, sessionUrl.find('?'));
    if (!baseUrl.empty() && baseUrl[baseUrl.size() - 1] == '/')
    {
        baseUrl = baseUrl.substr(0, baseUrl.size() - 1);
    }
    webSocketUrl = baseUrl.c_str();
    webSocketUrl = webSocketUrl.replace("https://", "wss://").replace("http://", "ws://") + "/game";

    password = QUrlQuery(QUrl(sessionUrl.c_str()).query()).queryItemValue("pwd");
}

void Connect4Widget::handleGameFinishedEvent(const QString& result)
{
    if (!m_enabled)
    {
        return;
    }

    if (result == "winner")
    {
        addRotatingSinDesire(0, 255, 0);
    }
    else if (result == "loser")
    {
        addRotatingSinDesire(255, 0, 0);
    }
    else
    {
        addRotatingSinDesire(255, 255, 0);
    }
}

void Connect4Widget::addRotatingSinDesire(uint8_t r, uint8_t g, uint8_t b)
{
    constexpr double SPEED = 2.0;
    constexpr double DURATION_S = 4.0;

    daemon_ros_client::msg::LedColor c;
    c.red = r;
    c.green = g;
    c.blue = b;

    m_desireSet->addDesire<LedAnimationDesire>(
        "rotating_sin",
        std::vector<daemon_ros_client::msg::LedColor>{c},
        SPEED,
        DURATION_S);
}
