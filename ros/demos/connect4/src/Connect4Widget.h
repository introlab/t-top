#ifndef CONNECT4_CONNECT4_WIDGET_H
#define CONNECT4_CONNECT4_WIDGET_H

#include "ImageDisplay.h"

#include <QWidget>
#include <QTimer>

#include <ros/ros.h>
#include <hbba_lite/core/DesireSet.h>
#include <std_msgs/Empty.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>

#include <atomic>
#include <memory>

class Connect4Widget : public QWidget
{
    Q_OBJECT
    ros::NodeHandle& m_nodeHandle;
    std::shared_ptr<DesireSet> m_desireSet;

    std::atomic_bool m_enabled;

    ros::Subscriber m_startButtonPressedSub;
    ros::Subscriber m_stopButtonPressedSub;
    ros::Subscriber m_remoteImageSub;

    ros::Publisher  m_volumePub;
    QTimer* m_setVolumeTimer;

public:
    Connect4Widget(ros::NodeHandle& nodeHandle, std::shared_ptr<DesireSet> desireSet, QWidget* parent = nullptr);

private slots:
    void onSetVolumeTimerTimeout();

private:
    void startButtonPressedCallback(const std_msgs::EmptyConstPtr& msg);
    void stopButtonPressedCallback(const std_msgs::EmptyConstPtr& msg);
    void remoteImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg);

    void setVolume(float volume);

private:
    ImageDisplay* m_imageDisplay;
};
#endif
