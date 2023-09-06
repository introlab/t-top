#include "Connect4Widget.h"
#include "QtUtils.h"

#include <t_top_hbba_lite/Desires.h>
#include <std_msgs/Float32.h>

#include <QVBoxLayout>

constexpr float ENABLED_VOLUME = 1;
constexpr float DISABLED_VOLUME = 0;
constexpr int SET_VOLUME_TIMER_INTERVAL_MS = 500;

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
        QImage image(&msg->frame.data[0], msg->frame.width, msg->frame.height, QImage::Format_RGB888);
        invokeLater([this, image]() { m_imageDisplay->setImage(image.rgbSwapped()); });
    }
}

void Connect4Widget::setVolume(float volume)
{
    std_msgs::Float32 msg;
    msg.data = volume;
    m_volumePub.publish(msg);
}
