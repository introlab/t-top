#include "Connect4Widget.h"
#include "QtUtils.h"

#include <t_top_hbba_lite/Desires.h>
#include <std_msgs/Float32.h>

#include <QVBoxLayout>
#include <QUrl>
#include <QUrlQuery>
#include <QJsonDocument>
#include <QJsonObject>

constexpr float ENABLED_VOLUME = 1;
constexpr float DISABLED_VOLUME = 0;
constexpr int SET_VOLUME_TIMER_INTERVAL_MS = 500;

using namespace opentera;

std::string openteraClientGetClientType(const Client& client)
{
    if (client.data()->get_flag() != sio::message::flag_object)
    {
        return "";
    }

    auto typeIt = client.data()->get_map().find("type");
    if (typeIt == client.data()->get_map().end())
    {
        return "";
    }

    if (typeIt->second->get_flag() != sio::message::flag_string)
    {
        return "";
    }

    return typeIt->second->get_string();
}


Connect4Widget::Connect4Widget(ros::NodeHandle& nodeHandle, std::shared_ptr<DesireSet> desireSet, QWidget* parent)
    : m_nodeHandle(nodeHandle),
      m_desireSet(std::move(desireSet)),
      m_enabled(false)
{
    m_imageDisplay = new ImageDisplay;
    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(m_imageDisplay);
    setLayout(layout);

    m_startButtonPressedSub =
        m_nodeHandle.subscribe("daemon/start_button_pressed", 1, &Connect4Widget::startButtonPressedCallback, this);
    m_stopButtonPressedSub =
        m_nodeHandle.subscribe("daemon/stop_button_pressed", 1, &Connect4Widget::stopButtonPressedCallback, this);
    m_remoteImageSub = m_nodeHandle.subscribe("/webrtc_image", 1, &Connect4Widget::remoteImageCallback, this);

    m_volumePub = m_nodeHandle.advertise<std_msgs::Float32>("volume", 1);

    m_setVolumeTimer = new QTimer(this);
    connect(m_setVolumeTimer, &QTimer::timeout, this, &Connect4Widget::onSetVolumeTimerTimeout);
    m_setVolumeTimer->start(SET_VOLUME_TIMER_INTERVAL_MS);

    m_openteraEventSubscriber = m_nodeHandle.subscribe("events", 10, &Connect4Widget::openteraEventCallback, this);
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

void Connect4Widget::startButtonPressedCallback(const std_msgs::EmptyConstPtr& msg)
{
    m_enabled = true;
    setVolume(ENABLED_VOLUME);
    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire<NearestFaceFollowingDesire>();
    m_desireSet->addDesire<Camera3dRecordingDesire>();
}

void Connect4Widget::stopButtonPressedCallback(const std_msgs::EmptyConstPtr& msg)
{
    m_enabled = false;
    setVolume(DISABLED_VOLUME);
    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->removeAllDesiresOfType<NearestFaceFollowingDesire>();
    m_desireSet->removeAllDesiresOfType<Camera3dRecordingDesire>();
    m_desireSet->removeAllDesiresOfType<LedAnimationDesire>();
    invokeLater([this]() { m_imageDisplay->setImage(QImage()); });
}

void Connect4Widget::remoteImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg)
{
    if (msg->frame.encoding != "bgr8")
    {
        ROS_ERROR("Not support image encoding (Connect4Widget::remoteImageCallback).");
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
    std_msgs::Float32 msg;
    msg.data = volume;
    m_volumePub.publish(msg);
}

void Connect4Widget::openteraEventCallback(const opentera_webrtc_ros_msgs::OpenTeraEventConstPtr& msg)
{
    if (!msg->join_session_events.empty())
    {
        std::string deviceName = msg->current_device_name;
        std::string sessionUrl = msg->join_session_events[0].session_url;
        std::string sessionParameters = msg->join_session_events[0].session_parameters;
        invokeLater([=]() { connectGameDataChannel(deviceName, sessionUrl, sessionParameters); });
    }
    if (!msg->stop_session_events.empty())
    {
        invokeLater([this]() { closeGameDataChannel(); });
    }
}

void Connect4Widget::connectGameDataChannel(
    const std::string& deviceName,
    const std::string& sessionUrl,
    const std::string& sessionParameters)
{
    constexpr bool VERIFY_TLS = true;

    m_deviceName = deviceName;
    m_participantName = getParticipantName(deviceName, sessionParameters);

    std::string baseUrl;
    std::string password;
    parseSessionUrl(sessionUrl, baseUrl, password);

    std::vector<IceServer> iceServers;
    if (!IceServer::fetchFromServer(baseUrl + "/iceservers", password, iceServers, VERIFY_TLS))
    {
        ROS_ERROR("Connect4 Game Data Channel: IceServer::fetchFromServer failed");
        iceServers.clear();
    }

    auto clientData = sio::object_message::create();
    clientData->get_map()["type"] = sio::string_message::create("robot");

    auto webrtcConfiguration = WebrtcConfiguration::create(iceServers);
    auto dataChannelConfiguration = DataChannelConfiguration::create();
    auto signalingServerConfiguration = SignalingServerConfiguration::create(baseUrl + "/socket.io", deviceName, clientData, "game", password);

    m_gameDataChannelClient = std::make_unique<DataChannelClient>(signalingServerConfiguration, webrtcConfiguration, dataChannelConfiguration);
    m_gameDataChannelClient->setTlsVerificationEnabled(VERIFY_TLS);

    setGameDataChannelCallbacks();

    ROS_INFO_STREAM("Connect4 Game Data Channel: Connection (sessionUrl=" << sessionUrl <<
        ", baseUrl=" << baseUrl <<
        ", password=" << password <<
        ", deviceName=" << m_deviceName <<
        ", participantName" << m_participantName << ")");
    m_gameDataChannelClient->connect();
}

std::string Connect4Widget::getParticipantName(const std::string& deviceName, const std::string& sessionParameters)
{
    QJsonParseError jsonParseError;
    QJsonDocument jsonMessage = QJsonDocument::fromJson(QString(sessionParameters.c_str()).toUtf8(), &jsonParseError);

    if (jsonParseError.error != QJsonParseError::NoError)
    {
        ROS_ERROR_STREAM("Connect4 Game Data Channel: getParticipantName error: " << jsonParseError.errorString().toStdString());
        return "";
    }

    QJsonObject jsonObject = jsonMessage.object();
    QString participant1 = jsonObject["participant1"].toString();
    QString participant2 = jsonObject["partiticipant2"].toString();
    QString robot1 = jsonObject["robot1"].toString();
    QString robot2 = jsonObject["robot2"].toString();

    QString qDeviceName(deviceName.c_str());
    if (qDeviceName == robot1)
    {
        return participant1.toStdString();
    }
    else if (qDeviceName == robot2)
    {
        return participant2.toStdString();
    }
    else
    {
        ROS_ERROR("Connect4 Game Data Channel: Device name not found");
        return "";
    }
}

void Connect4Widget::parseSessionUrl(const std::string& sessionUrl, std::string& baseUrl, std::string& password)
{
    baseUrl = sessionUrl.substr(0, sessionUrl.find('?'));
    if (!baseUrl.empty() && baseUrl[baseUrl.size() - 1] == '/')
    {
        baseUrl = baseUrl.substr(0, baseUrl.size() - 1);
    }
    password = QUrlQuery(QUrl(sessionUrl.c_str()).query()).queryItemValue("pwd").toStdString();
}

void Connect4Widget::setGameDataChannelCallbacks()
{
    m_gameDataChannelClient->setOnSignalingConnectionOpened([]()
        {
            ROS_INFO("Connect4 Game Data Channel: Signaling Connection Opened");
        });
    m_gameDataChannelClient->setOnSignalingConnectionClosed([]()
        {
            ROS_INFO("Connect4 Game Data Channel: Signaling Connection Closed");
        });
    m_gameDataChannelClient->setOnSignalingConnectionError([](const std::string& error)
        {
            ROS_ERROR_STREAM("Connect4 Game Data Channel: Signaling Connection Error : " << error);
        });

    m_gameDataChannelClient->setCallAcceptor([](const Client& client)
        {
            return openteraClientGetClientType(client) == "manager";
        });

    m_gameDataChannelClient->setOnDataChannelOpened([](const Client& client)
        {
            ROS_INFO_STREAM("Connect4 Game Data Channel: Data Channel Opened : " << client.id() << ", " << client.name());
        });
    m_gameDataChannelClient->setOnDataChannelClosed([](const Client& client)
        {
            ROS_INFO_STREAM("Connect4 Game Data Channel: Data Channel Closed : " << client.id() << ", " << client.name());
        });
    m_gameDataChannelClient->setOnDataChannelError([](const Client& client, const std::string& error)
        {
            ROS_ERROR_STREAM("Connect4 Game Data Channel: Data Channel Closed : " << client.id() << ", " << client.name() << " : " << error);
        });
    m_gameDataChannelClient->setOnDataChannelMessageString([this](const Client& client, const std::string& message)
        {
            invokeLater([this, message]() { handleGameMessage(message.c_str()); });
        });

    m_gameDataChannelClient->setOnError([](const std::string& error)
        {
            ROS_ERROR_STREAM("Connect4 Game Data Channel: Error : " << error);
        });
    m_gameDataChannelClient->setLogger([](const std::string& message)
        {
            ROS_ERROR_STREAM("Connect4 Game Data Channel: Log : " << message);
        });
}

void Connect4Widget::handleGameMessage(const QString& message)
{
    if (!m_enabled)
    {
        return;
    }

    QStringList values = message.split(" ");

    if (values[0] == "game_finished" && values.size() == 2)
    {
        if (values[1] == "tie")
        {
            addRotatingSinDesire(255, 255, 0);
        }
        else if (isWinner(values[1].toStdString()))
        {
            addRotatingSinDesire(0, 255, 0);
        }
        else
        {
            addRotatingSinDesire(255, 0, 0);
        }
    }
}

bool Connect4Widget::isWinner(const std::string& participantId)
{
    return m_gameDataChannelClient->getRoomClient(participantId).name() == m_participantName;
}

void Connect4Widget::addRotatingSinDesire(uint8_t r, uint8_t g, uint8_t b)
{
    constexpr double SPEED = 2.0;
    constexpr double DURATION_S = 4.0;

    daemon_ros_client::LedColor c;
    c.red = r;
    c.green = g;
    c.blue = b;

    m_desireSet->addDesire<LedAnimationDesire>(
        "rotating_sin",
        std::vector<daemon_ros_client::LedColor>{c},
        SPEED,
        DURATION_S);
}

void Connect4Widget::closeGameDataChannel()
{
    m_deviceName = "";
    m_participantName = "";
    m_gameDataChannelClient.reset();
}
